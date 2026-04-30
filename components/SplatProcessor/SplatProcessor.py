import torch
import numpy as np
import math
from scipy.spatial.transform import Rotation
from sharp.utils.gaussians import Gaussians3D, apply_transform
from datatype import View
from components.SplatProcessor.utils import (
    bilinear_interpolate_grid,
    project_gaussians_to_2d,
    compute_per_point_scales,
    project_world_cloud_to_view,
    rotate_to_pose,
    trim_by_fov,
    trim_by_max_depth,
    merge
)

class SplatProcessor:
    MAX_DEPTH = 20.0  # metres; increase to keep more background, decrease to cut sky/far objects

    def __init__(self, grid_resolution=8, detail_weight=0.0):
        self.grid_resolution = grid_resolution
        self.detail_weight = detail_weight

    def _build_scale_grid(
        self,
        px_ok: np.ndarray,
        py_ok: np.ndarray,
        raw_scale_ok: np.ndarray,
        grid_cells_x: int,
        grid_cells_y: int,
        image_width: int,
        image_height: int,
        median_scale: float
    ):
        """Step 3: Distribute point scales into a grid and compute cell medians."""
        cell_width = image_width / grid_cells_x
        cell_height = image_height / grid_cells_y
        grid = np.full((grid_cells_y, grid_cells_x), median_scale, dtype=np.float32)
        
        for gy in range(grid_cells_y):
            for gx in range(grid_cells_x):
                in_cell = ((px_ok >= gx * cell_width) & (px_ok < (gx + 1) * cell_width) &
                           (py_ok >= gy * cell_height) & (py_ok < (gy + 1) * cell_height))
                if int(in_cell.sum()) >= 8:
                    cell_scales = raw_scale_ok[in_cell]
                    cl, ch = np.quantile(cell_scales, [0.1, 0.9])
                    cell_trimmed = cell_scales[(cell_scales >= cl) & (cell_scales <= ch)]
                    if cell_trimmed.size > 0:
                        grid[gy, gx] = float(np.median(cell_trimmed))

        return np.clip(grid, median_scale * 0.1, median_scale * 10.0), cell_width, cell_height

    def _apply_smooth_alignment(
        self,
        gaussians: Gaussians3D,
        grid: np.ndarray,
        pixel_x: np.ndarray,
        pixel_y: np.ndarray,
        cell_width: float,
        cell_height: float,
        grid_cells_x: int,
        grid_cells_y: int,
        image_width: int,
        image_height: int,
        raw_scale_ok: np.ndarray,
        valid: np.ndarray,
        ok: np.ndarray,
        median_scale: float
    ):
        """Step 4: Interpolate grid scales and apply to Gaussian parameters."""
        all_px = np.clip(pixel_x, 0, image_width - 1)
        all_py = np.clip(pixel_y, 0, image_height - 1)
        smooth_scale = bilinear_interpolate_grid(
            grid, all_px, all_py, cell_width, cell_height, grid_cells_x, grid_cells_y
        )

        dw = float(np.clip(self.detail_weight, 0.0, 1.0))
        if dw > 0.0:
            per_point_raw = np.full(pixel_x.shape[0], median_scale, dtype=np.float32)
            valid_indices = np.where(valid)[0]
            ok_within_valid = np.where(ok)[0]
            per_point_raw[valid_indices[ok_within_valid]] = raw_scale_ok
            per_point_scale = smooth_scale * (1.0 - dw) + per_point_raw * dw
        else:
            per_point_scale = smooth_scale

        device = gaussians.mean_vectors.device
        scale_tensor = torch.tensor(per_point_scale, dtype=torch.float32, device=device).unsqueeze(1)
        
        return Gaussians3D(
            mean_vectors=gaussians.mean_vectors * scale_tensor,
            singular_values=gaussians.singular_values * scale_tensor,
            quaternions=gaussians.quaternions,
            colors=gaussians.colors,
            opacities=gaussians.opacities,
        )

    def align_gaussians_to_depth(
        self,
        gaussians: Gaussians3D,
        reference_depth: np.ndarray,
        focal_x_px: float,
        focal_y_px: float,
        image_width: int,
        image_height: int,
    ) -> Gaussians3D:
        """Align Gaussian depths to a reference depth map using a smooth scale grid."""
        # 1. Projection (from utils)
        pixel_x, pixel_y, depth_z, radial, valid = project_gaussians_to_2d(
            gaussians, focal_x_px, focal_y_px, image_width, image_height
        )
        
        num_valid = int(valid.sum())
        if num_valid < 64:
            return gaussians

        # 2. Compute point scales (from utils)
        raw_scale_ok, median_scale, ok = compute_per_point_scales(
            pixel_x, pixel_y, radial, depth_z, reference_depth, valid, use_radial=True
        )
        
        if raw_scale_ok is None:
            return gaussians

        # 3. Build grid (Local logic)
        grid_cells_x = max(1, int(self.grid_resolution))
        grid_cells_y = max(1, int(round(grid_cells_x * (image_height / max(1, image_width)))))
        
        grid, cell_width, cell_height = self._build_scale_grid(
            pixel_x[valid][ok], pixel_y[valid][ok], raw_scale_ok,
            grid_cells_x, grid_cells_y, image_width, image_height, median_scale
        )

        # 4. Apply smooth alignment (Local logic)
        return self._apply_smooth_alignment(
            gaussians, grid, pixel_x, pixel_y, cell_width, cell_height,
            grid_cells_x, grid_cells_y, image_width, image_height,
            raw_scale_ok, valid, ok, median_scale
        )

    def process(self, views: list[View], splats_list: list[Gaussians3D], pano_poses: dict = None, da3_world_pts: np.ndarray = None) -> Gaussians3D:
        """Main processing loop: align, trim, pose, and merge."""
        processed_splats = []

        for view, splat in zip(views, splats_list):
            R_local = Rotation.from_euler('yx', [view.yaw, view.pitch], degrees=True).as_matrix()
            pano_data = pano_poses.get(view.pano_id) if pano_poses else None
            center = pano_data['center'] if pano_data else None
            pano_rot = pano_data['rotation'] if pano_data else None
            # Full C2W rotation: pano_rot.T rotates from panorama-local into DA3 world
            R_c2w = pano_rot.T @ R_local if pano_rot is not None else R_local

            # 1. Depth alignment
            # ref_depth = None
            # if da3_world_pts is not None and center is not None:
            #     ref_depth = project_world_cloud_to_view(da3_world_pts, center, R_c2w, view)
            # elif view.depth is not None:
            #     ref_depth = view.depth

            # if ref_depth is not None:
            #     splat = self.align_gaussians_to_depth(
            #         splat, ref_depth, view.focal_px, view.focal_px, int(view.width), int(view.height)
            #     )

            # 2. Trim edges and far points (must happen before pose, while still in camera space)
            splat = trim_by_fov(splat, hfov_limit=view.hfov - 6.0)
            splat = trim_by_max_depth(splat, self.MAX_DEPTH)

            # 3. Apply global pose (C2W)
            if center is not None:
                c2w = np.eye(4)
                c2w[:3, :3] = R_c2w
                c2w[:3, 3] = center
                splat = apply_transform(splat, torch.tensor(c2w[:3, :], dtype=torch.float32, device=splat.mean_vectors.device))
            else:
                splat = rotate_to_pose(splat, yaw=view.yaw, pitch=view.pitch)

            processed_splats.append(splat)

        return merge(processed_splats)
