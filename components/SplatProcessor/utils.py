import torch
import numpy as np
import math
from scipy.spatial.transform import Rotation
from sharp.utils.gaussians import Gaussians3D, apply_transform

def bilinear_interpolate_grid(grid, all_px, all_py, cell_width, cell_height, grid_cells_x, grid_cells_y):
    """Generic bilinear interpolation for a 2D grid."""
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

def project_gaussians_to_2d(
    gaussians: Gaussians3D, 
    focal_x_px: float, 
    focal_y_px: float, 
    image_width: int, 
    image_height: int
):
    """Project 3D Gaussians to 2D pixel coordinates."""
    mv_np = gaussians.mean_vectors[0].detach().cpu().numpy().astype(np.float32)
    depth_z = mv_np[:, 2]
    radial = np.linalg.norm(mv_np, axis=1)
    
    # Avoid division by zero
    safe_z = np.clip(depth_z, 1e-6, None)
    pixel_x = (mv_np[:, 0] / safe_z) * focal_x_px + (image_width / 2.0) - 0.5
    pixel_y = (mv_np[:, 1] / safe_z) * focal_y_px + (image_height / 2.0) - 0.5
    
    valid = (depth_z > 1e-6)
    valid &= (pixel_x >= 0) & (pixel_x <= image_width - 1)
    valid &= (pixel_y >= 0) & (pixel_y <= image_height - 1)
    
    return pixel_x, pixel_y, depth_z, radial, valid

def compute_per_point_scales(
    pixel_x: np.ndarray,
    pixel_y: np.ndarray,
    radial: np.ndarray,
    depth_z: np.ndarray,
    reference_depth: np.ndarray,
    valid: np.ndarray,
    use_radial: bool = True
):
    """Calculate the scale ratio between reference depth and current splat depth."""
    px_int = np.clip(np.round(pixel_x[valid]).astype(np.int32), 0, reference_depth.shape[1] - 1)
    py_int = np.clip(np.round(pixel_y[valid]).astype(np.int32), 0, reference_depth.shape[0] - 1)
    ref_depth_sampled = reference_depth[py_int, px_int]
    
    current_depth = radial[valid] if use_radial else depth_z[valid]
    ok = np.isfinite(ref_depth_sampled) & (ref_depth_sampled > 1e-6) & (current_depth > 1e-6)
    
    if int(ok.sum()) < 64:
        return None, 1.0, None

    ref_depth_ok = ref_depth_sampled[ok].astype(np.float32)
    current_depth_ok = current_depth[ok]
    raw_scale_ok = ref_depth_ok / current_depth_ok

    lo, hi = np.quantile(raw_scale_ok, [0.05, 0.95])
    trimmed = raw_scale_ok[(raw_scale_ok >= lo) & (raw_scale_ok <= hi)]
    median_scale = float(np.median(trimmed)) if trimmed.size > 0 else float(np.median(raw_scale_ok))
    
    return raw_scale_ok, median_scale, ok

def rotate_to_pose(gaussians: Gaussians3D, yaw: float, pitch: float) -> Gaussians3D:
    """Manually rotates a splat based on yaw/pitch."""
    device = gaussians.mean_vectors.device
    rotation = Rotation.from_euler('yx', [yaw, pitch], degrees=True)
    transform = torch.cat([
        torch.tensor(rotation.as_matrix(), dtype=torch.float32).to(device), 
        torch.zeros((3, 1)).to(device)
    ], dim=1)
    return apply_transform(gaussians, transform)

def trim_by_fov(gaussians: Gaussians3D, hfov_limit: float) -> Gaussians3D:
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

def merge(splats_list: list[Gaussians3D]) -> Gaussians3D:
    """Merges a list of Gaussians3D objects into one."""
    if not splats_list:
        return None
    return Gaussians3D(
        mean_vectors=torch.cat([item.mean_vectors for item in splats_list], dim=1),
        singular_values=torch.cat([item.singular_values for item in splats_list], dim=1),
        quaternions=torch.cat([item.quaternions for item in splats_list], dim=1),
        colors=torch.cat([item.colors for item in splats_list], dim=1),
        opacities=torch.cat([item.opacities for item in splats_list], dim=1),
    )
