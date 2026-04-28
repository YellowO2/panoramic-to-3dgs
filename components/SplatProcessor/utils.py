import torch
import numpy as np
import math
import cv2
import os
from typing import Tuple, Optional
from scipy.spatial.transform import Rotation
from sharp.utils.gaussians import Gaussians3D, apply_transform

def backproject_views_to_pcd(views: list, da3_result):
    """
    Back-projects processed views into world space for debugging.
    Uses the snapped extrinsics directly from DA3 result.
    """
    all_points = []
    all_colors = []
    
    pred = da3_result.prediction
    if pred is None: return None, None

    for i, v in enumerate(views):
        # 1. Geometry from DA3
        K = pred.intrinsics[i]
        depth = pred.depth[i]
        # Use the extrinsics we already snapped in DA3Model
        w2c = pred.extrinsics[i]
        conf = pred.conf[i] if pred.conf is not None else None
        
        h, w = depth.shape
        us, vs = np.meshgrid(np.arange(w), np.arange(h))
        pix = np.stack([us, vs, np.ones_like(us)], axis=-1).reshape(-1, 3)
        
        valid = np.isfinite(depth) & (depth > 0)
        if conf is not None:
            valid &= (conf >= np.percentile(conf, 40))
        
        vidx = np.flatnonzero(valid.reshape(-1))
        if len(vidx) == 0: continue
        
        # 2. Backproject to Camera Space
        K_inv = np.linalg.inv(K)
        rays = (K_inv @ pix[vidx].T).T
        pts_cam = rays * depth.flatten()[vidx][:, None]
        
        # 3. Transform to World Space (using C2W)
        w2c_homo = np.eye(4)
        w2c_homo[:3, :4] = w2c[:3, :4]
        c2w = np.linalg.inv(w2c_homo)
        
        pts_world = (c2w[:3, :3] @ pts_cam.T).T + c2w[:3, 3]
        all_points.append(pts_world)
        
        # 4. Colors
        if v.path and os.path.exists(v.path):
            img_bgr = cv2.imread(v.path)
            if img_bgr is not None:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                if img_rgb.shape[:2] != (h, w): img_rgb = cv2.resize(img_rgb, (w, h))
                all_colors.append(img_rgb.reshape(-1, 3)[vidx] / 255.0)
            
    if not all_points: return None, None
    return np.concatenate(all_points, axis=0), (np.concatenate(all_colors, axis=0) if all_colors else None)

def panoramic_depth_to_pcd(
    depth: np.ndarray, 
    image: Optional[np.ndarray] = None, 
    v_fov_deg: float = None, 
    max_depth_mult: float = 5.0
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Converts a panoramic depth map to a point cloud."""
    h, w = depth.shape
    if v_fov_deg is None:
        theta_start, theta_end = 0.0, np.pi
    else:
        v_fov_rad = np.radians(v_fov_deg)
        theta_start, theta_end = (np.pi / 2.0) - (v_fov_rad / 2.0), (np.pi / 2.0) + (v_fov_rad / 2.0)

    theta = np.linspace(theta_start, theta_end, h)
    theta_grid = np.repeat(theta.reshape(h, 1), w, axis=1)
    phi = np.linspace(-np.pi, np.pi, w)
    phi_grid = np.repeat(phi.reshape(1, w), h, axis=0)

    x = depth * np.sin(theta_grid) * np.sin(phi_grid)
    y = depth * np.cos(theta_grid)
    z = depth * np.sin(theta_grid) * np.cos(phi_grid)
    points = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)

    colors_filtered = None
    if image is not None:
        img_resized = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        colors_filtered = img_rgb.reshape(-1, 3) / 255.0

    d_flat = depth.flatten()
    valid_mask = d_flat > 1e-3
    if np.any(valid_mask):
        mask = valid_mask & (d_flat < (np.median(d_flat[valid_mask]) * max_depth_mult))
    else: mask = valid_mask
        
    return points[mask], (colors_filtered[mask] if colors_filtered is not None else None)

def project_world_cloud_to_view(world_pts: np.ndarray, center: np.ndarray, R_local: np.ndarray, view) -> np.ndarray:
    """
    Projects a world-space point cloud into a single view's depth buffer.
    Returns a (H, W) depth map in metres; 0 means no data.
    Minimum-Z (closest surface) wins when multiple points land on the same pixel.
    """
    R_w2c = R_local.T
    pts_cam = (R_w2c @ (world_pts - center).T).T  # (N, 3)

    valid = pts_cam[:, 2] > 0.1
    pts_cam = pts_cam[valid]
    if len(pts_cam) == 0:
        return np.zeros((int(view.height), int(view.width)), dtype=np.float32)

    u = (pts_cam[:, 0] / pts_cam[:, 2]) * view.focal_px + view.width / 2.0
    v = (pts_cam[:, 1] / pts_cam[:, 2]) * view.focal_px + view.height / 2.0
    z = pts_cam[:, 2]

    in_bounds = (u >= 0) & (u < view.width) & (v >= 0) & (v < view.height)
    ui = np.round(u[in_bounds]).astype(np.int32)
    vi = np.round(v[in_bounds]).astype(np.int32)
    zi = z[in_bounds]

    depth_map = np.full((int(view.height), int(view.width)), np.inf, dtype=np.float32)
    np.minimum.at(depth_map, (vi, ui), zi)
    depth_map[depth_map == np.inf] = 0.0
    return depth_map


def bilinear_interpolate_grid(grid, all_px, all_py, cell_width, cell_height, grid_cells_x, grid_cells_y):
    gx_cont, gy_cont = all_px / cell_width - 0.5, all_py / cell_height - 0.5
    gx0, gy0 = np.clip(np.floor(gx_cont).astype(np.int32), 0, grid_cells_x - 1), np.clip(np.floor(gy_cont).astype(np.int32), 0, grid_cells_y - 1)
    gx1, gy1 = np.clip(gx0 + 1, 0, grid_cells_x - 1), np.clip(gy0 + 1, 0, grid_cells_y - 1)
    wx, wy = np.clip(gx_cont - gx0, 0, 1).astype(np.float32), np.clip(gy_cont - gy0, 0, 1).astype(np.float32)
    s00, s01, s10, s11 = grid[gy0, gx0], grid[gy0, gx1], grid[gy1, gx0], grid[gy1, gx1]
    return (s00 * (1 - wx) * (1 - wy) + s01 * wx * (1 - wy) + s10 * (1 - wx) * wy + s11 * wx * wy)

def project_gaussians_to_2d(gaussians: Gaussians3D, focal_x_px: float, focal_y_px: float, image_width: int, image_height: int):
    mv_np = gaussians.mean_vectors[0].detach().cpu().numpy().astype(np.float32)
    depth_z = mv_np[:, 2]
    radial, safe_z = np.linalg.norm(mv_np, axis=1), np.clip(depth_z, 1e-6, None)
    pixel_x = (mv_np[:, 0] / safe_z) * focal_x_px + (image_width / 2.0) - 0.5
    pixel_y = (mv_np[:, 1] / safe_z) * focal_y_px + (image_height / 2.0) - 0.5
    valid = (depth_z > 1e-6) & (pixel_x >= 0) & (pixel_x <= image_width - 1) & (pixel_y >= 0) & (pixel_y <= image_height - 1)
    return pixel_x, pixel_y, depth_z, radial, valid

def compute_per_point_scales(pixel_x, pixel_y, radial, depth_z, reference_depth, valid, use_radial=True):
    px_int, py_int = np.clip(np.round(pixel_x[valid]).astype(np.int32), 0, reference_depth.shape[1] - 1), np.clip(np.round(pixel_y[valid]).astype(np.int32), 0, reference_depth.shape[0] - 1)
    ref_depth_sampled, current_depth = reference_depth[py_int, px_int], (radial[valid] if use_radial else depth_z[valid])
    ok = np.isfinite(ref_depth_sampled) & (ref_depth_sampled > 1e-6) & (current_depth > 1e-6)
    if int(ok.sum()) < 64: return None, 1.0, None
    raw_scale_ok = ref_depth_sampled[ok].astype(np.float32) / current_depth[ok]
    lo, hi = np.quantile(raw_scale_ok, [0.05, 0.95])
    trimmed = raw_scale_ok[(raw_scale_ok >= lo) & (raw_scale_ok <= hi)]
    return raw_scale_ok, float(np.median(trimmed)) if trimmed.size > 0 else float(np.median(raw_scale_ok)), ok

def rotate_to_pose(gaussians: Gaussians3D, yaw: float, pitch: float) -> Gaussians3D:
    rotation = Rotation.from_euler('yx', [yaw, pitch], degrees=True)
    transform = torch.cat([torch.tensor(rotation.as_matrix(), dtype=torch.float32).to(gaussians.mean_vectors.device), torch.zeros((3, 1)).to(gaussians.mean_vectors.device)], dim=1)
    return apply_transform(gaussians, transform)

def trim_by_fov(gaussians, hfov_limit):
    positions = gaussians.mean_vectors
    x, z = positions[0][:, 0], positions[0][:, 2]
    mask = (z > 0) & (torch.abs(x / z) <= math.tan(math.radians(hfov_limit / 2.0)))
    return Gaussians3D(mean_vectors=gaussians.mean_vectors[:, mask, :], singular_values=gaussians.singular_values[:, mask, :], quaternions=gaussians.quaternions[:, mask, :], colors=gaussians.colors[:, mask, :], opacities=gaussians.opacities[:, mask])

def merge(splats_list: list[Gaussians3D]) -> Gaussians3D:
    if not splats_list: return None
    return Gaussians3D(mean_vectors=torch.cat([item.mean_vectors for item in splats_list], dim=1), singular_values=torch.cat([item.singular_values for item in splats_list], dim=1), quaternions=torch.cat([item.quaternions for item in splats_list], dim=1), colors=torch.cat([item.colors for item in splats_list], dim=1), opacities=torch.cat([item.opacities for item in splats_list], dim=1))
