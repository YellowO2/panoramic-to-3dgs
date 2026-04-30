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
    scale_gaussians,
    measure_nearest_z,
    trim_by_fov,
    trim_by_max_depth,
    merge,
    apply_smooth_alignment,
)


class SplatProcessor:
    MAX_DEPTH = 15.0  # metres; increase to keep more background, decrease to cut sky/far objects

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
        median_scale: float,
    ):
        """Step 3: Distribute point scales into a grid and compute cell medians."""
        cell_width = image_width / grid_cells_x
        cell_height = image_height / grid_cells_y
        grid = np.full((grid_cells_y, grid_cells_x), median_scale, dtype=np.float32)

        for gy in range(grid_cells_y):
            for gx in range(grid_cells_x):
                in_cell = (
                    (px_ok >= gx * cell_width)
                    & (px_ok < (gx + 1) * cell_width)
                    & (py_ok >= gy * cell_height)
                    & (py_ok < (gy + 1) * cell_height)
                )
                if int(in_cell.sum()) >= 8:
                    cell_scales = raw_scale_ok[in_cell]
                    cl, ch = np.quantile(cell_scales, [0.1, 0.9])
                    cell_trimmed = cell_scales[
                        (cell_scales >= cl) & (cell_scales <= ch)
                    ]
                    if cell_trimmed.size > 0:
                        grid[gy, gx] = float(np.median(cell_trimmed))

        return (
            np.clip(grid, median_scale * 0.1, median_scale * 10.0),
            cell_width,
            cell_height,
        )

    # ====== I commented these alignment algo out cause they SUCKED!! =====
    # def align_gaussians_to_depth(
    #     self,
    #     gaussians: Gaussians3D,
    #     reference_depth: np.ndarray,
    #     focal_x_px: float,
    #     focal_y_px: float,
    #     image_width: int,
    #     image_height: int,
    # ) -> Gaussians3D:
    #     """Align Gaussian depths to a reference depth map using a smooth scale grid."""
    #     # 1. Projection (from utils)
    #     pixel_x, pixel_y, depth_z, radial, valid = project_gaussians_to_2d(
    #         gaussians, focal_x_px, focal_y_px, image_width, image_height
    #     )

    #     num_valid = int(valid.sum())
    #     if num_valid < 64:
    #         return gaussians

    #     # 2. Compute point scales (from utils)
    #     raw_scale_ok, median_scale, ok = compute_per_point_scales(
    #         pixel_x, pixel_y, radial, depth_z, reference_depth, valid, use_radial=True
    #     )

    #     if raw_scale_ok is None:
    #         return gaussians

    #     # 3. Build grid (Local logic)
    #     grid_cells_x = max(1, int(self.grid_resolution))
    #     grid_cells_y = max(1, int(round(grid_cells_x * (image_height / max(1, image_width)))))

    #     grid, cell_width, cell_height = self._build_scale_grid(
    #         pixel_x[valid][ok], pixel_y[valid][ok], raw_scale_ok,
    #         grid_cells_x, grid_cells_y, image_width, image_height, median_scale
    #     )

    #     # 4. Apply smooth alignment (Local logic)
    #     return self._apply_smooth_alignment(
    #         gaussians, grid, pixel_x, pixel_y, cell_width, cell_height,
    #         grid_cells_x, grid_cells_y, image_width, image_height,
    #         raw_scale_ok, valid, ok, median_scale
    #     )

    # def align_gaussians_global_scale_da3(
    #     self,
    #     gaussians: Gaussians3D,
    #     reference_depth: np.ndarray,
    #     focal_x_px: float,
    #     focal_y_px: float,
    #     image_width: int,
    #     image_height: int,
    # ) -> Gaussians3D:
    #     """Scale all Gaussians by a single global median scale derived from the reference depth."""
    #     pixel_x, pixel_y, depth_z, radial, valid = project_gaussians_to_2d(
    #         gaussians, focal_x_px, focal_y_px, image_width, image_height
    #     )
    #     if int(valid.sum()) < 64:
    #         return gaussians

    #     # use_radial=False because reference_depth stores Z (camera-plane depth)
    #     raw_scale_ok, median_scale, ok = compute_per_point_scales(
    #         pixel_x, pixel_y, radial, depth_z, reference_depth, valid, use_radial=False
    #     )
    #     if raw_scale_ok is None or median_scale <= 0:
    #         return gaussians
    #     print(f"Global scale factor: {median_scale:.3f}")
    #     return scale_gaussians(gaussians, median_scale)
    # ==== End of alignment algo that sucked ===

    def align_gaussians_per_point(
        self,
        gaussians: Gaussians3D,
        reference_depth: np.ndarray,
        focal_x_px: float,
        focal_y_px: float,
        image_width: int,
        image_height: int,
    ) -> Gaussians3D:
        """Scale each Gaussian by its own DA3 depth ratio; median scale for unmatched points."""
        pixel_x, pixel_y, depth_z, radial, valid = project_gaussians_to_2d(
            gaussians, focal_x_px, focal_y_px, image_width, image_height
        )
        if int(valid.sum()) < 64:
            return gaussians

        raw_scale_ok, median_scale, ok = compute_per_point_scales(
            pixel_x, pixel_y, radial, depth_z, reference_depth, valid, use_radial=False
        )
        if raw_scale_ok is None or median_scale <= 0:
            return gaussians

        per_point_scale = np.full(pixel_x.shape[0], median_scale, dtype=np.float32)
        valid_indices = np.where(valid)[0]
        ok_indices = np.where(ok)[0]
        per_point_scale[valid_indices[ok_indices]] = raw_scale_ok

        device = gaussians.mean_vectors.device
        scale_tensor = torch.tensor(per_point_scale, dtype=torch.float32, device=device).unsqueeze(1)
        return Gaussians3D(
            mean_vectors=gaussians.mean_vectors * scale_tensor,
            singular_values=gaussians.singular_values * scale_tensor,
            quaternions=gaussians.quaternions,
            colors=gaussians.colors,
            opacities=gaussians.opacities,
        )

    def align_splats_by_near_edge(
        self, views: list[View], splats_list: list[Gaussians3D]
    ) -> list[Gaussians3D]:
        """
        Scale each splat so all nearest-Z distances match the median across slices.
        Uses 0.1th percentile Z (≈ ground distance) as the scale proxy.
        Assumes trim_by_max_depth has already been applied (camera space, before pose).
        """
        nearest_zs = []
        for view, splat in zip(views, splats_list):
            z = measure_nearest_z(splat)
            nearest_zs.append(z)
            print(
                f"  Nearest Z [{view.yaw:+.0f}°]: {f'{z:.3f}' if z is not None else 'N/A'}"
            )

        valid_zs = [z for z in nearest_zs if z is not None]
        if not valid_zs:
            print("Near edge alignment: no valid measurements, skipping.")
            return splats_list

        target = float(np.median(valid_zs))
        print(f"Near edge target Z: {target:.3f}")

        result = []
        for splat, z in zip(splats_list, nearest_zs):
            if z is not None and z > 1e-6:
                result.append(scale_gaussians(splat, target / z))
            else:
                result.append(splat)
        return result

    def process(
        self,
        views: list[View],
        splats_list: list[Gaussians3D],
        pano_poses: dict = None,
        da3_world_pts: np.ndarray = None,
        scale_mode: str = "da3",
    ) -> Gaussians3D:
        """Main processing loop: align, trim, pose, and merge.
        scale_mode: 'da3' uses DA3 world cloud projection; 'bottom_edge' uses bottom-edge width matching.
        """
        processed_splats = []

        # Pre-compute per-view poses (needed for da3 mode inside loop)
        view_poses = []
        for view in views:
            R_local = Rotation.from_euler(
                "yx", [view.yaw, view.pitch], degrees=True
            ).as_matrix()
            pano_data = pano_poses.get(view.pano_id) if pano_poses else None
            center = pano_data["center"] if pano_data else None
            pano_rot = pano_data["rotation"] if pano_data else None
            R_c2w = pano_rot.T @ R_local if pano_rot is not None else R_local
            view_poses.append((R_local, center, pano_rot, R_c2w))

        # Step 1: trim far points first (camera space)
        trimmed = [trim_by_max_depth(splat, self.MAX_DEPTH) for splat in splats_list]

        # Step 2: scale alignment
        if scale_mode == "near_edge":
            print("--- Scale mode: near edge width ---")
            trimmed = self.align_splats_by_near_edge(views, trimmed)
        else:
            print("--- Scale mode: DA3 projection ---")
            for i, (view, splat) in enumerate(zip(views, trimmed)):
                _, center, _, R_c2w = view_poses[i]
                ref_depth = None
                pano_pts = da3_world_pts.get(view.pano_id) if isinstance(da3_world_pts, dict) else da3_world_pts
                if pano_pts is not None and center is not None:
                    ref_depth = project_world_cloud_to_view(
                        pano_pts, center, R_c2w, view
                    )
                elif view.depth is not None:
                    ref_depth = view.depth
                if ref_depth is not None:
                    trimmed[i] = self.align_gaussians_per_point(
                        splat,
                        ref_depth,
                        view.focal_px,
                        view.focal_px,
                        int(view.width),
                        int(view.height),
                    )

        # Step 3: trim FOV edges, then apply pose
        for i, (view, splat) in enumerate(zip(views, trimmed)):
            _, center, _, R_c2w = view_poses[i]
            splat = trim_by_fov(splat, hfov_limit=view.hfov - 6.0)

            if center is not None:
                c2w = np.eye(4)
                c2w[:3, :3] = R_c2w
                c2w[:3, 3] = center
                splat = apply_transform(
                    splat,
                    torch.tensor(
                        c2w[:3, :],
                        dtype=torch.float32,
                        device=splat.mean_vectors.device,
                    ),
                )
            else:
                splat = rotate_to_pose(splat, yaw=view.yaw, pitch=view.pitch)

            processed_splats.append(splat)

        return merge(processed_splats)
