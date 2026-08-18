import torch
import numpy as np
import math
import cv2
import os
from scipy.spatial.transform import Rotation
from sharp.utils.gaussians import Gaussians3D, apply_transform

# DA3's own conf output is `1 + exp(x)` (unbounded, not a [0,1] probability,
# not calibrated across scenes) -- see model/utils/head_utils.py's
# "expp1" activation. Matches DA3's own reference export (utils/export/glb.py
# get_conf_thresh), just stricter on the percentile: absolute floor, clamped
# between a lower and upper percentile so a uniformly low-confidence view
# still gets filtered (floor) but a call never discards everything (upper
# clamp).
CONF_ABS_FLOOR = 1.05
CONF_LOWER_PERCENTILE = 50.0
CONF_UPPER_PERCENTILE = 90.0


def _wedge_bounds_per_pano(views: list) -> dict:
    """For each pano's surviving views, the angular wedge (yaw degrees,
    relative to that view's OWN yaw) it "owns" in the final point cloud --
    a circular Voronoi split of the pano's 360 degrees by view-center yaw,
    computed only from views actually present here (so a view dropped by
    the consensus filter isn't a factor: its neighbors' wedges just expand
    to cover the gap it left, automatically).

    Adjacent views are typically 20-30 degrees apart with a 90 degree HFOV,
    so without this, the same real-world stretch of wall/curb gets
    contributed by 2-3 different views' depth estimates that don't quite
    agree pixel-for-pixel -- inflating point count and adding a soft
    "doubled up" fuzziness. Trimming each view to only its own wedge (its
    least-distorted, most-central region, since perspective crops stretch
    more toward the edges) means every real-world direction comes from
    exactly one view.

    Only handles yaw (not pitch) -- every view this pipeline actually
    generates is sliced at pitch=0 (see extract_views_for_da3), so a 1D
    circular partition is sufficient; there's no multi-row case to cover.

    Returns {view_index: (lo_deg, hi_deg)}, bounds expressed relative to
    that view's own yaw, directly comparable to a per-pixel yaw offset
    computed from that same view's own camera rays -- no absolute-angle
    wraparound bookkeeping needed."""
    by_pano = {}
    for i, v in enumerate(views):
        by_pano.setdefault(v.pano_id, []).append((i, v.yaw % 360.0))

    bounds = {}
    for pano_id, entries in by_pano.items():
        entries.sort(key=lambda e: e[1])
        n = len(entries)
        for k in range(n):
            idx, yaw = entries[k]
            if n == 1:
                bounds[idx] = (-180.0, 180.0)
                continue
            prev_yaw = entries[k - 1][1] - (360.0 if k == 0 else 0.0)
            next_yaw = entries[(k + 1) % n][1] + (360.0 if k == n - 1 else 0.0)
            bounds[idx] = ((prev_yaw - yaw) / 2.0, (next_yaw - yaw) / 2.0)
    return bounds


def backproject_views_to_pcd(views: list, da3_result):
    """
    Back-projects processed views into world space.
    Returns (all_pts, all_cols) combined, plus per_pano dicts
    {pano_id: pts} and {pano_id: colors}.

    `views` must be index-aligned with da3_result.prediction (i.e. the exact
    list DA3 was run on, not an arbitrary subset/reorder) — depth/pose/color
    are looked up positionally by enumerate(views).

    Each view only contributes points from its own angular wedge (see
    _wedge_bounds_per_pano) -- not its whole overlapping field of view.
    """
    all_points = []
    all_colors = []
    per_pano_pts: dict[int, list] = {}
    per_pano_cols: dict[int, list] = {}

    pred = da3_result.prediction
    if pred is None:
        return None, None, {}, {}

    wedge_bounds = _wedge_bounds_per_pano(views)

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

        # Per-pixel yaw offset from this view's own optical axis (verified:
        # atan2(ray_x, ray_z) on K_inv @ pixel recovers the same yaw
        # convention extract_views_for_da3's THETA uses -- 0 at image
        # center, +/-HFOV/2 at the left/right edges), used to keep only
        # this view's own wedge.
        K_inv = np.linalg.inv(K)
        rays_full = (K_inv @ pix.T).T
        yaw_offset = np.degrees(np.arctan2(rays_full[:, 0], rays_full[:, 2])).reshape(h, w)
        lo, hi = wedge_bounds.get(i, (-180.0, 180.0))
        in_wedge = (yaw_offset >= lo) & (yaw_offset < hi)

        valid = np.isfinite(depth) & (depth > 0) & in_wedge
        if conf is not None:
            lower = np.percentile(conf, CONF_LOWER_PERCENTILE)
            upper = np.percentile(conf, CONF_UPPER_PERCENTILE)
            conf_thr = min(max(CONF_ABS_FLOOR, lower), upper)
            valid &= conf >= conf_thr

        vidx = np.flatnonzero(valid.reshape(-1))
        if len(vidx) == 0:
            continue

        # 2. Backproject to Camera Space
        rays = rays_full[vidx]
        pts_cam = rays * depth.flatten()[vidx][:, None]

        # 3. Transform to World Space (using C2W)
        w2c_homo = np.eye(4)
        w2c_homo[:3, :4] = w2c[:3, :4]
        c2w = np.linalg.inv(w2c_homo)

        pts_world = (c2w[:3, :3] @ pts_cam.T).T + c2w[:3, 3]
        all_points.append(pts_world)
        per_pano_pts.setdefault(v.pano_id, []).append(pts_world)

        # 4. Colors
        if v.path and os.path.exists(v.path):
            img_bgr = cv2.imread(v.path)
            if img_bgr is not None:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                if img_rgb.shape[:2] != (h, w):
                    img_rgb = cv2.resize(img_rgb, (w, h))
                cols = img_rgb.reshape(-1, 3)[vidx] / 255.0
                all_colors.append(cols)
                per_pano_cols.setdefault(v.pano_id, []).append(cols)

    if not all_points:
        return None, None, {}, {}
    consolidated_pts = {pid: np.concatenate(pts, axis=0) for pid, pts in per_pano_pts.items()}
    consolidated_cols = {pid: np.concatenate(cols, axis=0) for pid, cols in per_pano_cols.items()}
    return (
        np.concatenate(all_points, axis=0),
        np.concatenate(all_colors, axis=0) if all_colors else None,
        consolidated_pts,
        consolidated_cols,
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


def measure_nearest_z(gaussians: Gaussians3D) -> float | None:
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


def split_depth_zones(
    gaussians: Gaussians3D,
    align_depth: float,
    near_depth: float,
    sky_depth: float,
) -> tuple[Gaussians3D, Gaussians3D, Gaussians3D]:
    """Single-pass split into three zones: align (≤align_depth), keep (align_depth, near_depth], sky (>sky_depth)."""
    radial = torch.norm(gaussians.mean_vectors[0], dim=1)
    m_align = radial <= align_depth
    m_keep  = (radial > align_depth) & (radial <= near_depth)
    m_sky   = radial > sky_depth

    def _select(mask):
        return Gaussians3D(
            mean_vectors=gaussians.mean_vectors[:, mask, :],
            singular_values=gaussians.singular_values[:, mask, :],
            quaternions=gaussians.quaternions[:, mask, :],
            colors=gaussians.colors[:, mask, :],
            opacities=gaussians.opacities[:, mask],
        )

    return _select(m_align), _select(m_keep), _select(m_sky)


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


def subsample_gaussians(gaussians: Gaussians3D, keep_fraction: float) -> Gaussians3D:
    if keep_fraction >= 1.0:
        return gaussians
    n = gaussians.mean_vectors.shape[1]
    keep = max(1, int(n * keep_fraction))
    idx = torch.randperm(n, device=gaussians.mean_vectors.device)[:keep]
    return Gaussians3D(
        mean_vectors=gaussians.mean_vectors[:, idx, :],
        singular_values=gaussians.singular_values[:, idx, :],
        quaternions=gaussians.quaternions[:, idx, :],
        colors=gaussians.colors[:, idx, :],
        opacities=gaussians.opacities[:, idx],
    )


def trim_by_cone(gaussians: Gaussians3D, half_angle_deg: float) -> Gaussians3D:
    """Keep only Gaussians within a cone of half_angle_deg around the camera Z-axis (forward/down)."""
    positions = gaussians.mean_vectors[0]
    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
    radial_xy = torch.sqrt(x ** 2 + y ** 2)
    limit = math.tan(math.radians(half_angle_deg))
    mask = (z > 0) & (radial_xy / z.clamp(min=1e-6) <= limit)
    return Gaussians3D(
        mean_vectors=gaussians.mean_vectors[:, mask, :],
        singular_values=gaussians.singular_values[:, mask, :],
        quaternions=gaussians.quaternions[:, mask, :],
        colors=gaussians.colors[:, mask, :],
        opacities=gaussians.opacities[:, mask],
    )


def trim_by_pitch_bottom(gaussians: Gaussians3D, max_down_deg: float) -> Gaussians3D:
    """Trim Gaussians below max_down_deg downward pitch (Y-down camera convention)."""
    positions = gaussians.mean_vectors[0]
    y, z = positions[:, 1], positions[:, 2]
    limit = math.tan(math.radians(max_down_deg))
    mask = (z > 0) & (y / z.clamp(min=1e-6) <= limit)
    return Gaussians3D(
        mean_vectors=gaussians.mean_vectors[:, mask, :],
        singular_values=gaussians.singular_values[:, mask, :],
        quaternions=gaussians.quaternions[:, mask, :],
        colors=gaussians.colors[:, mask, :],
        opacities=gaussians.opacities[:, mask],
    )


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


