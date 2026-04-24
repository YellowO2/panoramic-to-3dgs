import torch
import numpy as np
import math
import cv2
from scipy.spatial.transform import Rotation
from sharp.utils.gaussians import Gaussians3D, apply_transform
from datatype import View

class SplatProcessor:
    def __init__(self, grid_resolution=8, detail_weight=0.0):
        self.grid_resolution = grid_resolution
        self.detail_weight = detail_weight

    def _bilinear_interpolate_grid(self, grid, all_px, all_py, cell_width, cell_height, grid_cells_x, grid_cells_y):
        gx_cont = all_px / cell_width - 0.5
        gy_cont = all_py / cell_height - 0.5
        gx0 = np.clip(np.floor(gx_cont).astype(np.int32), 0, grid_cells_x - 1)
        gy0 = np.clip(np.floor(gy_cont).astype(np.int32), 0, grid_cells_y - 1)
        gx1 = np.clip(gx0 + 1, 0, grid_cells_x - 1)
        gy1 = np.clip(gy0 + 1, 0, grid_cells_y - 1)
        wx = np.clip(gx_cont - gx0, 0, 1).astype(np.float32)
        wy = np.clip(gy_cont - gy0, 0, 1).astype(np.float32)
        s00 = grid[gy0, gx0]
        s01 = grid[gy0, gx1]
        s10 = grid[gy1, gx0]
        s11 = grid[gy1, gx1]
        return (s00 * (1 - wx) * (1 - wy) + s01 * wx * (1 - wy) + 
                s10 * (1 - wx) * wy + s11 * wx * wy)

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
        grid_cells_x = max(1, int(self.grid_resolution))
        grid_cells_y = max(1, int(round(grid_cells_x * (image_height / max(1, image_width)))))

        mv_np = gaussians.mean_vectors[0].detach().cpu().numpy().astype(np.float32)
        depth_z = mv_np[:, 2]
        radial = np.linalg.norm(mv_np, axis=1)
        
        valid = depth_z > 1e-6
        pixel_x = (mv_np[:, 0] / np.clip(depth_z, 1e-6, None)) * focal_x_px + (image_width / 2.0) - 0.5
        pixel_y = (mv_np[:, 1] / np.clip(depth_z, 1e-6, None)) * focal_y_px + (image_height / 2.0) - 0.5
        valid &= (pixel_x >= 0) & (pixel_x <= image_width - 1)
        valid &= (pixel_y >= 0) & (pixel_y <= image_height - 1)

        per_point_scale = np.ones(mv_np.shape[0], dtype=np.float32)
        median_scale = 1.0

        if int(valid.sum()) >= 64:
            px_int = np.clip(np.round(pixel_x[valid]).astype(np.int32), 0, image_width - 1)
            py_int = np.clip(np.round(pixel_y[valid]).astype(np.int32), 0, image_height - 1)
            ref_depth_sampled = reference_depth[py_int, px_int]
            
            ok = np.isfinite(ref_depth_sampled) & (ref_depth_sampled > 1e-6) & (radial[valid] > 1e-6)
            if int(ok.sum()) >= 64:
                ref_depth_ok = ref_depth_sampled[ok].astype(np.float32)
                sharp_r_ok = radial[valid][ok]
                raw_scale = ref_depth_ok / sharp_r_ok

                lo, hi = np.quantile(raw_scale, [0.05, 0.95])
                trimmed = raw_scale[(raw_scale >= lo) & (raw_scale <= hi)]
                median_scale = float(np.median(trimmed)) if trimmed.size > 0 else float(np.median(raw_scale))

                # Build grid
                px_ok = pixel_x[valid][ok]
                py_ok = pixel_y[valid][ok]
                cell_width = image_width / grid_cells_x
                cell_height = image_height / grid_cells_y
                grid = np.full((grid_cells_y, grid_cells_x), median_scale, dtype=np.float32)
                
                for gy in range(grid_cells_y):
                    for gx in range(grid_cells_x):
                        in_cell = ((px_ok >= gx * cell_width) & (px_ok < (gx + 1) * cell_width) &
                                   (py_ok >= gy * cell_height) & (py_ok < (gy + 1) * cell_height))
                        if int(in_cell.sum()) >= 8:
                            cell_scales = raw_scale[in_cell]
                            cl, ch = np.quantile(cell_scales, [0.1, 0.9])
                            cell_trimmed = cell_scales[(cell_scales >= cl) & (cell_scales <= ch)]
                            if cell_trimmed.size > 0:
                                grid[gy, gx] = float(np.median(cell_trimmed))

                grid = np.clip(grid, median_scale * 0.1, median_scale * 10.0)
                all_px = np.clip(pixel_x, 0, image_width - 1)
                all_py = np.clip(pixel_y, 0, image_height - 1)
                smooth_scale = self._bilinear_interpolate_grid(grid, all_px, all_py, cell_width, cell_height, grid_cells_x, grid_cells_y)

                dw = float(np.clip(self.detail_weight, 0.0, 1.0))
                if dw > 0.0:
                    per_point_raw = np.full(mv_np.shape[0], median_scale, dtype=np.float32)
                    valid_indices = np.where(valid)[0]
                    ok_within_valid = np.where(ok)[0]
                    per_point_raw[valid_indices[ok_within_valid]] = raw_scale
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

    def trim_by_fov(self, gaussians: Gaussians3D, hfov_limit: float) -> Gaussians3D:
        """Trims dirty edges of the splat based on field of view."""
        positions = gaussians.mean_vectors
        x = positions[0][:, 0]
        z = positions[0][:, 2]
        half_hfov_rad = math.radians(hfov_limit / 2.0)
        max_x_ratio = math.tan(half_hfov_rad)
        mask = (z > 0) & (torch.abs(x / z) <= max_x_ratio)
        
        return Gaussians3D(
            mean_vectors=gaussians.mean_vectors[:, mask, :],
            singular_values=gaussians.singular_values[:, mask, :],
            quaternions=gaussians.quaternions[:, mask, :],
            colors=gaussians.colors[:, mask, :],
            opacities=gaussians.opacities[:, mask],
        )

    def rotate_to_pose(self, gaussians: Gaussians3D, yaw: float, pitch: float) -> Gaussians3D:
        """Manually rotates a splat based on yaw/pitch."""
        device = gaussians.mean_vectors.device
        rotation = Rotation.from_euler('yx', [yaw, pitch], degrees=True)
        transform = torch.cat([
            torch.tensor(rotation.as_matrix(), dtype=torch.float32).to(device), 
            torch.zeros((3, 1)).to(device)
        ], dim=1)
        return apply_transform(gaussians, transform)

    def merge(self, splats_list: list[Gaussians3D]) -> Gaussians3D:
        """Merges a list of Gaussians3D objects into one."""
        return Gaussians3D(
            mean_vectors=torch.cat([item.mean_vectors for item in splats_list], dim=1),
            singular_values=torch.cat([item.singular_values for item in splats_list], dim=1),
            quaternions=torch.cat([item.quaternions for item in splats_list], dim=1),
            colors=torch.cat([item.colors for item in splats_list], dim=1),
            opacities=torch.cat([item.opacities for item in splats_list], dim=1),
        )

    def process(self, views: list[View], splats_list: list[Gaussians3D], enable_alignment: bool = True) -> Gaussians3D:
        """Main processing loop: align, trim, pose, and merge."""
        processed_splats = []
        
        for i, (view, splat) in enumerate(zip(views, splats_list)):
            # 1. Alignment (if depth map exists in View)
            if enable_alignment and view.depth is not None:
                focal_px = float(view.focal_px)
                splat = self.align_gaussians_to_depth(
                    splat, view.depth, focal_px, focal_px, int(view.width), int(view.height)
                )
                print(f"Aligned splat {i+1}/{len(views)} using reference depth.")

            # 2. Trim edges
            hfov_keep = view.hfov - 6.0
            splat = self.trim_by_fov(splat, hfov_limit=hfov_keep)
            
            # 3. Apply Global Pose
            if enable_alignment and view.extrinsics is not None:
                # Use extrinsics provided by DA3
                w2c = view.extrinsics
                r_c2w = w2c[:, :3].T
                t_c2w = -r_c2w @ w2c[:, 3:]
                c2w = np.hstack((r_c2w, t_c2w))
                device = splat.mean_vectors.device
                c2w_tensor = torch.tensor(c2w, dtype=torch.float32, device=device)
                splat = apply_transform(splat, c2w_tensor)
            else:
                # Fallback to manual rotation (e.g. for DA360 panoramic views)
                splat = self.rotate_to_pose(splat, yaw=view.yaw, pitch=view.pitch)

            processed_splats.append(splat)
            
        return self.merge(processed_splats)
