import torch
import numpy as np
import math
import cv2
import os
from typing import Tuple, Optional
from scipy.spatial.transform import Rotation
from scipy.ndimage import gaussian_filter1d
from sharp.utils.gaussians import Gaussians3D, apply_transform


def backproject_views_to_pcd(views: list, da3_result):
    """
    Back-projects processed views into world space.
    Returns (all_pts, all_cols) combined, plus per_pano dict {pano_id: pts}.
    """
    all_points = []
    all_colors = []
    per_pano: dict[int, list] = {}

    pred = da3_result.prediction
    if pred is None:
        return None, None, {}

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
            valid &= conf >= np.percentile(conf, 40)

        vidx = np.flatnonzero(valid.reshape(-1))
        if len(vidx) == 0:
            continue

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
        per_pano.setdefault(v.pano_id, []).append(pts_world)

        # 4. Colors
        if v.path and os.path.exists(v.path):
            img_bgr = cv2.imread(v.path)
            if img_bgr is not None:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                if img_rgb.shape[:2] != (h, w):
                    img_rgb = cv2.resize(img_rgb, (w, h))
                all_colors.append(img_rgb.reshape(-1, 3)[vidx] / 255.0)

    if not all_points:
        return None, None, {}
    consolidated = {pid: np.concatenate(pts, axis=0) for pid, pts in per_pano.items()}
    return np.concatenate(all_points, axis=0), (
        np.concatenate(all_colors, axis=0) if all_colors else None
    ), consolidated


def panoramic_depth_to_pcd(
    depth: np.ndarray,
    image: Optional[np.ndarray] = None,
    v_fov_deg: float = None,
    max_depth_mult: float = 5.0,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Converts a panoramic depth map to a point cloud."""
    h, w = depth.shape
    if v_fov_deg is None:
        theta_start, theta_end = 0.0, np.pi
    else:
        v_fov_rad = np.radians(v_fov_deg)
        theta_start, theta_end = (np.pi / 2.0) - (v_fov_rad / 2.0), (np.pi / 2.0) + (
            v_fov_rad / 2.0
        )

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
    else:
        mask = valid_mask

    return points[mask], (
        colors_filtered[mask] if colors_filtered is not None else None
    )


def project_world_cloud_to_view(
    world_pts: np.ndarray, center: np.ndarray, R_c2w: np.ndarray, view, max_depth: float = None
) -> np.ndarray:
    """
    Projects a world-space point cloud into a single view's depth buffer.
    R_c2w is the full camera-to-world rotation (pano_rot.T @ R_local).
    Returns a (H, W) depth map in metres; 0 means no data.
    Minimum-Z (closest surface) wins when multiple points land on the same pixel.
    max_depth: if set, discards points with camera-space Z beyond this value (filters cross-wall leakage).
    """
    R_w2c = R_c2w.T
    pts_cam = (R_w2c @ (world_pts - center).T).T  # (N, 3)

    valid = pts_cam[:, 2] > 0.1
    if max_depth is not None:
        valid &= pts_cam[:, 2] <= max_depth
    pts_cam = pts_cam[valid]
    if len(pts_cam) == 0:
        return np.zeros((int(view.height), int(view.width)), dtype=np.float32)

    u = (pts_cam[:, 0] / pts_cam[:, 2]) * view.focal_px + view.width / 2.0
    v = (pts_cam[:, 1] / pts_cam[:, 2]) * view.focal_px + view.height / 2.0
    z = pts_cam[:, 2]

    in_bounds = (u >= 0) & (u < view.width) & (v >= 0) & (v < view.height)
    ui = np.clip(np.round(u[in_bounds]).astype(np.int32), 0, int(view.width) - 1)
    vi = np.clip(np.round(v[in_bounds]).astype(np.int32), 0, int(view.height) - 1)
    zi = z[in_bounds]

    depth_map = np.full((int(view.height), int(view.width)), np.inf, dtype=np.float32)
    np.minimum.at(depth_map, (vi, ui), zi)
    depth_map[depth_map == np.inf] = 0.0
    return depth_map


def bilinear_interpolate_grid(
    grid, all_px, all_py, cell_width, cell_height, grid_cells_x, grid_cells_y
):
    gx_cont, gy_cont = all_px / cell_width - 0.5, all_py / cell_height - 0.5
    gx0, gy0 = np.clip(
        np.floor(gx_cont).astype(np.int32), 0, grid_cells_x - 1
    ), np.clip(np.floor(gy_cont).astype(np.int32), 0, grid_cells_y - 1)
    gx1, gy1 = np.clip(gx0 + 1, 0, grid_cells_x - 1), np.clip(
        gy0 + 1, 0, grid_cells_y - 1
    )
    wx, wy = np.clip(gx_cont - gx0, 0, 1).astype(np.float32), np.clip(
        gy_cont - gy0, 0, 1
    ).astype(np.float32)
    s00, s01, s10, s11 = grid[gy0, gx0], grid[gy0, gx1], grid[gy1, gx0], grid[gy1, gx1]
    return (
        s00 * (1 - wx) * (1 - wy)
        + s01 * wx * (1 - wy)
        + s10 * (1 - wx) * wy
        + s11 * wx * wy
    )


def project_gaussians_to_2d(
    gaussians: Gaussians3D,
    focal_x_px: float,
    focal_y_px: float,
    image_width: int,
    image_height: int,
):
    mv_np = gaussians.mean_vectors[0].detach().cpu().numpy().astype(np.float32)
    depth_z = mv_np[:, 2]
    radial, safe_z = np.linalg.norm(mv_np, axis=1), np.clip(depth_z, 1e-6, None)
    pixel_x = (mv_np[:, 0] / safe_z) * focal_x_px + (image_width / 2.0) - 0.5
    pixel_y = (mv_np[:, 1] / safe_z) * focal_y_px + (image_height / 2.0) - 0.5
    valid = (
        (depth_z > 1e-6)
        & (pixel_x >= 0)
        & (pixel_x <= image_width - 1)
        & (pixel_y >= 0)
        & (pixel_y <= image_height - 1)
    )
    return pixel_x, pixel_y, depth_z, radial, valid


def compute_per_point_scales(
    pixel_x, pixel_y, radial, depth_z, reference_depth, valid, use_radial=True
):
    px_int, py_int = np.clip(
        np.round(pixel_x[valid]).astype(np.int32), 0, reference_depth.shape[1] - 1
    ), np.clip(
        np.round(pixel_y[valid]).astype(np.int32), 0, reference_depth.shape[0] - 1
    )
    ref_depth_sampled, current_depth = reference_depth[py_int, px_int], (
        radial[valid] if use_radial else depth_z[valid]
    )
    ok = (
        np.isfinite(ref_depth_sampled)
        & (ref_depth_sampled > 1e-6)
        & (current_depth > 1e-6)
    )
    if int(ok.sum()) < 64:
        return None, 1.0, None
    raw_scale_ok = ref_depth_sampled[ok].astype(np.float32) / current_depth[ok]
    lo, hi = np.quantile(raw_scale_ok, [0.10, 0.90])
    trimmed = raw_scale_ok[(raw_scale_ok >= lo) & (raw_scale_ok <= hi)]
    return (
        raw_scale_ok,
        float(np.mean(trimmed)) if trimmed.size > 0 else float(np.mean(raw_scale_ok)),
        ok,
    )


def measure_nearest_z(gaussians: Gaussians3D) -> float:
    """
    Returns the 0.1th percentile Z of all Gaussians as a proxy for the nearest surface distance.
    In street panoramas the ground is almost always visible, so this ≈ ground distance,
    which should be consistent across slices when scales are correct.
    Returns None if too few points.
    """
    mv = gaussians.mean_vectors[0].detach().cpu().numpy()
    z = mv[:, 2]
    z = z[z > 0.01]
    if len(z) < 16:
        return None
    return float(np.percentile(z, 0.1))


def scale_gaussians(gaussians: Gaussians3D, scale: float) -> Gaussians3D:
    s = torch.tensor(scale, dtype=torch.float32, device=gaussians.mean_vectors.device)
    return Gaussians3D(
        mean_vectors=gaussians.mean_vectors * s,
        singular_values=gaussians.singular_values * s,
        quaternions=gaussians.quaternions,
        colors=gaussians.colors,
        opacities=gaussians.opacities,
    )


def rotate_to_pose(gaussians: Gaussians3D, yaw: float, pitch: float) -> Gaussians3D:
    rotation = Rotation.from_euler("yx", [yaw, pitch], degrees=True)
    transform = torch.cat(
        [
            torch.tensor(rotation.as_matrix(), dtype=torch.float32).to(
                gaussians.mean_vectors.device
            ),
            torch.zeros((3, 1)).to(gaussians.mean_vectors.device),
        ],
        dim=1,
    )
    return apply_transform(gaussians, transform)


def trim_by_max_depth(gaussians: Gaussians3D, max_depth: float) -> Gaussians3D:
    radial = torch.norm(gaussians.mean_vectors[0], dim=1)
    mask = radial <= max_depth
    return Gaussians3D(
        mean_vectors=gaussians.mean_vectors[:, mask, :],
        singular_values=gaussians.singular_values[:, mask, :],
        quaternions=gaussians.quaternions[:, mask, :],
        colors=gaussians.colors[:, mask, :],
        opacities=gaussians.opacities[:, mask],
    )


def trim_by_fov(gaussians, hfov_limit):
    positions = gaussians.mean_vectors
    x, z = positions[0][:, 0], positions[0][:, 2]
    mask = (z > 0) & (torch.abs(x / z) <= math.tan(math.radians(hfov_limit / 2.0)))
    return Gaussians3D(
        mean_vectors=gaussians.mean_vectors[:, mask, :],
        singular_values=gaussians.singular_values[:, mask, :],
        quaternions=gaussians.quaternions[:, mask, :],
        colors=gaussians.colors[:, mask, :],
        opacities=gaussians.opacities[:, mask],
    )


def trim_by_pano_voronoi(
    gaussians: Gaussians3D,
    own_center: np.ndarray,
    other_centers: list,
    buffer_m: float = 1.0,
) -> Gaussians3D:
    """Keep Gaussians closer (XZ plane) to their own pano center than any other, plus a buffer."""
    if not other_centers:
        return gaussians
    mv = gaussians.mean_vectors[0].detach().cpu().numpy()
    xz = mv[:, [0, 2]]
    dist_own = np.linalg.norm(xz - own_center[[0, 2]], axis=1)
    other_dists = np.stack(
        [np.linalg.norm(xz - c[[0, 2]], axis=1) for c in other_centers], axis=1
    )
    min_other = other_dists.min(axis=1)
    mask = torch.tensor(dist_own <= min_other + buffer_m, device=gaussians.mean_vectors.device)
    return Gaussians3D(
        mean_vectors=gaussians.mean_vectors[:, mask, :],
        singular_values=gaussians.singular_values[:, mask, :],
        quaternions=gaussians.quaternions[:, mask, :],
        colors=gaussians.colors[:, mask, :],
        opacities=gaussians.opacities[:, mask],
    )


def correct_interpano_seams(
    per_pano_merged: dict,
    pano_centers: dict,
    boundary_band_m: float = 3.0,
    bin_width_m: float = 2.0,
    smooth_sigma_bins: float = 2.0,
) -> dict:
    """Shift near-boundary Gaussians to close inter-pano depth seams.

    For each pano pair (A, B):
      - Bins near-boundary Gaussians by their XZ position ALONG the Voronoi
        boundary line (the lateral dimension).
      - In each bin, computes the median depth of A and B perpendicular to the
        boundary (the direction where mismatch occurs).
      - Shifts both sides in the boundary-normal direction toward their midpoint.

    This avoids the ring artifact caused by using radial distance from the
    midpoint, which was the bug in the previous implementation.
    """
    pids = list(per_pano_merged.keys())
    if len(pids) < 2:
        return per_pano_merged

    positions = {
        pid: splat.mean_vectors[0].detach().cpu().numpy()
        for pid, splat in per_pano_merged.items()
    }
    shifts = {pid: np.zeros((len(positions[pid]), 3), dtype=np.float32) for pid in pids}

    def _fill_and_smooth(arr, sigma):
        valid = np.where(~np.isnan(arr))[0]
        if len(valid) == 0:
            return arr
        result = arr.copy()
        for k in np.where(np.isnan(arr))[0]:
            result[k] = arr[valid[np.argmin(np.abs(valid - k))]]
        if sigma > 0:
            result = gaussian_filter1d(result.astype(np.float64), sigma=sigma).astype(np.float32)
        return result

    for i in range(len(pids)):
        for j in range(i + 1, len(pids)):
            pid_A, pid_B = pids[i], pids[j]
            if pid_A not in pano_centers or pid_B not in pano_centers:
                continue

            cA = pano_centers[pid_A][[0, 2]].astype(np.float64)
            cB = pano_centers[pid_B][[0, 2]].astype(np.float64)
            sep = cB - cA
            sep_len = np.linalg.norm(sep)
            if sep_len < 1e-3:
                continue

            # Unit vectors: normal = A→B direction (depth axis), tangent = along boundary
            norm_dir = sep / sep_len        # perpendicular to Voronoi boundary
            tang_dir = np.array([-norm_dir[1], norm_dir[0]])  # along boundary

            xz_A = positions[pid_A][:, [0, 2]].astype(np.float64)
            xz_B = positions[pid_B][:, [0, 2]].astype(np.float64)

            # Near-boundary: |dist_to_A - dist_to_B| <= boundary_band_m
            dA_to_A = np.linalg.norm(xz_A - cA, axis=1)
            dA_to_B = np.linalg.norm(xz_A - cB, axis=1)
            dB_to_A = np.linalg.norm(xz_B - cA, axis=1)
            dB_to_B = np.linalg.norm(xz_B - cB, axis=1)

            idx_A = np.where(np.abs(dA_to_A - dA_to_B) <= boundary_band_m)[0]
            idx_B = np.where(np.abs(dB_to_A - dB_to_B) <= boundary_band_m)[0]

            print(f"  [Seam] Pano {pid_A}↔{pid_B}: {len(idx_A)} + {len(idx_B)} boundary Gaussians")
            if len(idx_A) < 4 or len(idx_B) < 4:
                continue

            mid_xz = (cA + cB) / 2.0
            rel_A = xz_A[idx_A] - mid_xz   # (n_A, 2)
            rel_B = xz_B[idx_B] - mid_xz   # (n_B, 2)

            # Project onto tangent (bin axis) and normal (shift axis)
            tang_A = rel_A @ tang_dir       # coord along boundary
            tang_B = rel_B @ tang_dir
            norm_A = rel_A @ norm_dir       # depth from midpoint (signed)
            norm_B = rel_B @ norm_dir

            # Bin along boundary tangent
            all_tang = np.concatenate([tang_A, tang_B])
            t_min = all_tang.min()
            num_bins = max(int((all_tang.max() - t_min) / bin_width_m) + 1, 1)

            bin_A = np.clip(((tang_A - t_min) / bin_width_m).astype(np.int32), 0, num_bins - 1)
            bin_B = np.clip(((tang_B - t_min) / bin_width_m).astype(np.int32), 0, num_bins - 1)

            med_norm_A = np.full(num_bins, np.nan, dtype=np.float32)
            med_norm_B = np.full(num_bins, np.nan, dtype=np.float32)
            for b in range(num_bins):
                m = bin_A == b
                if m.any():
                    med_norm_A[b] = float(np.median(norm_A[m]))
                m = bin_B == b
                if m.any():
                    med_norm_B[b] = float(np.median(norm_B[m]))

            both_valid = ~np.isnan(med_norm_A) & ~np.isnan(med_norm_B)
            if not both_valid.any():
                print(f"  [Seam] Pano {pid_A}↔{pid_B}: no shared tangent bins, skipping")
                continue

            # Target normal-coord: midpoint between A and B medians
            target_norm = np.where(both_valid, (med_norm_A + med_norm_B) / 2.0, np.nan).astype(np.float32)
            target_norm = _fill_and_smooth(target_norm, smooth_sigma_bins)

            # Shift each Gaussian in the normal direction to reach target
            delta_norm_A = (target_norm[bin_A] - norm_A).astype(np.float32)  # (n_A,)
            delta_norm_B = (target_norm[bin_B] - norm_B).astype(np.float32)  # (n_B,)

            delta_A_xz = delta_norm_A[:, None] * norm_dir.astype(np.float32)  # (n_A, 2)
            delta_B_xz = delta_norm_B[:, None] * norm_dir.astype(np.float32)  # (n_B, 2)

            shifts[pid_A][idx_A, 0] += delta_A_xz[:, 0]
            shifts[pid_A][idx_A, 2] += delta_A_xz[:, 1]
            shifts[pid_B][idx_B, 0] += delta_B_xz[:, 0]
            shifts[pid_B][idx_B, 2] += delta_B_xz[:, 1]

            print(
                f"  [Seam] Pano {pid_A}↔{pid_B}: mean shift "
                f"A={np.abs(delta_norm_A).mean():.3f}m  B={np.abs(delta_norm_B).mean():.3f}m"
            )

    # Apply accumulated shifts (XZ only, Y untouched)
    result = {}
    for pid, splat in per_pano_merged.items():
        sh = shifts[pid]
        if not np.any(sh):
            result[pid] = splat
            continue
        device = splat.mean_vectors.device
        sh_t = torch.tensor(sh, dtype=torch.float32, device=device)
        new_mv = splat.mean_vectors.clone()
        new_mv[0] += sh_t
        result[pid] = Gaussians3D(
            mean_vectors=new_mv,
            singular_values=splat.singular_values,
            quaternions=splat.quaternions,
            colors=splat.colors,
            opacities=splat.opacities,
        )
    return result


def merge(splats_list: list[Gaussians3D]) -> Gaussians3D:
    if not splats_list:
        return None
    return Gaussians3D(
        mean_vectors=torch.cat([item.mean_vectors for item in splats_list], dim=1),
        singular_values=torch.cat(
            [item.singular_values for item in splats_list], dim=1
        ),
        quaternions=torch.cat([item.quaternions for item in splats_list], dim=1),
        colors=torch.cat([item.colors for item in splats_list], dim=1),
        opacities=torch.cat([item.opacities for item in splats_list], dim=1),
    )


def apply_smooth_alignment(
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
    median_scale: float,
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
    scale_tensor = torch.tensor(
        per_point_scale, dtype=torch.float32, device=device
    ).unsqueeze(1)

    return Gaussians3D(
        mean_vectors=gaussians.mean_vectors * scale_tensor,
        singular_values=gaussians.singular_values * scale_tensor,
        quaternions=gaussians.quaternions,
        colors=gaussians.colors,
        opacities=gaussians.opacities,
    )
