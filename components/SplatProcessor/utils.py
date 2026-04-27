import torch
import numpy as np
import math
import cv2
import os
from typing import Tuple, Optional
from scipy.spatial.transform import Rotation
from sharp.utils.gaussians import Gaussians3D, apply_transform

def backproject_views_to_pcd(views: list, pano_poses: dict = None):
    """
    Back-projects multiple perspective views into a single world-space point cloud.
    """
    all_points = []
    all_colors = []
    
    for v in views:
        if v.depth is None: continue
        
        h, w = v.depth.shape
        x, y = np.arange(w), np.arange(h)
        xv, yv = np.meshgrid(x, y)
        
        cx, cy = (w / 2.0) - 0.5, (h / 2.0) - 0.5
        d = v.depth.flatten()
        valid = np.isfinite(d) & (d > 1e-4) & (d < 100.0)
        
        d_v, u_v, v_v = d[valid], xv.flatten()[valid], yv.flatten()[valid]
        
        # Camera Space
        pts_cam = np.stack([(u_v - cx) * d_v / v.focal_px, (v_v - cy) * d_v / v.focal_px, d_v], axis=1)
        
        # World Space
        if pano_poses and v.pano_id in pano_poses:
            center = pano_poses[v.pano_id]
            rot = Rotation.from_euler('yx', [v.yaw, v.pitch], degrees=True).as_matrix()
            pts_world = (rot @ pts_cam.T).T + center
        else:
            # Fallback to identity pose
            pts_world = pts_cam
            
        all_points.append(pts_world)
        
        # Colors
        img_rgb = None
        if v.path and os.path.exists(v.path):
            img_bgr = cv2.imread(v.path)
            if img_bgr is not None: img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
        if img_rgb is not None:
            if img_rgb.shape[:2] != (h, w): img_rgb = cv2.resize(img_rgb, (w, h))
            all_colors.append(img_rgb.reshape(-1, 3)[valid] / 255.0)
            
    if not all_points: return None, None
    return np.concatenate(all_points, axis=0), (np.concatenate(all_colors, axis=0) if all_colors else None)

def panoramic_depth_to_pcd(
    depth: np.ndarray, 
    image: Optional[np.ndarray] = None, 
    v_fov_deg: float = None, 
    max_depth_mult: float = 5.0
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Converts a panoramic depth map to anpoint cloud with colors. Sky is filtered out.
    """
    h, w = depth.shape
    
    # 1. Prepare Projection Math
    if v_fov_deg is None:
        theta_start, theta_end = 0.0, np.pi
    else:
        v_fov_rad = np.radians(v_fov_deg)
        theta_start = (np.pi / 2.0) - (v_fov_rad / 2.0)
        theta_end = (np.pi / 2.0) + (v_fov_rad / 2.0)

    theta = np.linspace(theta_start, theta_end, h)
    theta_grid = np.repeat(theta.reshape(h, 1), w, axis=1)
    phi = np.linspace(-np.pi, np.pi, w)
    phi_grid = np.repeat(phi.reshape(1, w), h, axis=0)

    # 2. Convert to XYZ
    x = depth * np.sin(theta_grid) * np.sin(phi_grid)
    y = depth * np.cos(theta_grid)
    z = depth * np.sin(theta_grid) * np.cos(phi_grid)
    points = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)

    # 3. Prepare Colors
    colors_filtered = None
    if image is not None:
        # Resize image to match depth map EXACTLY before flattening
        img_resized = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
        # Convert BGR to RGB if needed (assuming OpenCV input)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        colors_filtered = img_rgb.reshape(-1, 3) / 255.0

    # 4. Filter Sky & Invalid Points
    d_flat = depth.flatten()
    valid_mask = d_flat > 1e-3
    
    if np.any(valid_mask):
        median_d = np.median(d_flat[valid_mask])
        # Points too far (sky/noise) are removed
        mask = valid_mask & (d_flat < (median_d * max_depth_mult))
    else:
        mask = valid_mask
        
    points_filtered = points[mask]
    if colors_filtered is not None:
        colors_filtered = colors_filtered[mask]
        
    return points_filtered, colors_filtered

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
