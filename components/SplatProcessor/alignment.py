import math

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

from sharp.utils.gaussians import Gaussians3D
from components.SplatProcessor.utils import (
    measure_nearest_z,
    project_gaussians_to_2d,
    scale_gaussians,
)


def _voronoi_common(gaussians, reference_depth, focal_x_px, focal_y_px, image_width, image_height):
    """Shared Voronoi grid setup for DA3 alignment. Returns context dict or None on failure."""
    da3_v_full, da3_u_full = np.where(reference_depth > 0)
    n_anchors = len(da3_u_full)
    print(f"  [DA3] Anchors: {n_anchors}")
    if n_anchors < 4:
        return None

    pixel_x, pixel_y, depth_z, _, valid = project_gaussians_to_2d(
        gaussians, focal_x_px, focal_y_px, image_width, image_height
    )
    n_valid = int(valid.sum())
    print(f"  [DA3] Valid Gaussians: {n_valid} / {len(pixel_x)}")
    if n_valid < 16:
        return None

    aspect = image_width / image_height
    grid_w = max(int(min(math.sqrt(n_anchors * aspect), image_width)), 4)
    grid_h = max(int(min(math.sqrt(n_anchors / aspect), image_height)), 4)
    print(f"  [DA3] Grid: {grid_w}x{grid_h} (view: {image_width}x{image_height})")

    sx, sy = grid_w / image_width, grid_h / image_height
    da3_u_g = np.clip((da3_u_full * sx).astype(np.int32), 0, grid_w - 1)
    da3_v_g = np.clip((da3_v_full * sy).astype(np.int32), 0, grid_h - 1)
    grid_depth = np.zeros((grid_h, grid_w), dtype=np.float32)
    np.maximum.at(grid_depth, (da3_v_g, da3_u_g), reference_depth[da3_v_full, da3_u_full])

    dist_map, nearest = distance_transform_edt(grid_depth == 0, return_indices=True)
    max_dist_grid = math.sqrt(grid_w**2 + grid_h**2) / 4.0

    valid_idx = np.where(valid)[0]
    valid_px_g = np.clip((pixel_x[valid_idx] * sx).astype(np.int32), 0, grid_w - 1)
    valid_py_g = np.clip((pixel_y[valid_idx] * sy).astype(np.int32), 0, grid_h - 1)
    valid_dz = np.clip(depth_z[valid_idx], 1e-6, None)

    anchor_row = nearest[0, valid_py_g, valid_px_g]
    anchor_col = nearest[1, valid_py_g, valid_px_g]
    ref_z = grid_depth[anchor_row, anchor_col].astype(np.float32)
    raw_scales = ref_z / valid_dz
    gauss_dists = dist_map[valid_py_g, valid_px_g]
    within = gauss_dists <= max_dist_grid

    global_median = float(np.median(raw_scales[within])) if within.any() else 1.0
    print(f"  [DA3] Global median scale: {global_median:.4f}")
    if global_median <= 0:
        return None

    return dict(
        pixel_x=pixel_x,
        valid_idx=valid_idx,
        valid_dz=valid_dz,
        anchor_row=anchor_row,
        anchor_col=anchor_col,
        grid_w=grid_w,
        raw_scales=raw_scales,
        within=within,
        global_median=global_median,
    )


def _apply_per_gauss_scale(gaussians: Gaussians3D, per_gauss_scale: np.ndarray) -> Gaussians3D:
    device = gaussians.mean_vectors.device
    scale_tensor = torch.tensor(per_gauss_scale, dtype=torch.float32, device=device).unsqueeze(1)
    return Gaussians3D(
        mean_vectors=gaussians.mean_vectors * scale_tensor,
        singular_values=gaussians.singular_values * scale_tensor,
        quaternions=gaussians.quaternions,
        colors=gaussians.colors,
        opacities=gaussians.opacities,
    )


def align_near_edge(views, splats_list: list[Gaussians3D]) -> list[Gaussians3D]:
    """Scale each splat so all nearest-Z distances match the median across slices."""
    nearest_zs = [measure_nearest_z(splat) for splat in splats_list]
    for view, z in zip(views, nearest_zs):
        print(f"  Nearest Z [{view.yaw:+.0f}°]: {f'{z:.3f}' if z is not None else 'N/A'}")

    valid_zs = [z for z in nearest_zs if z is not None]
    if not valid_zs:
        print("  Near edge: no valid measurements, skipping.")
        return splats_list

    target = float(np.median(valid_zs))
    print(f"  Near edge target Z: {target:.3f}")
    return [
        scale_gaussians(splat, target / z) if (z is not None and z > 1e-6) else splat
        for splat, z in zip(splats_list, nearest_zs)
    ]


def align_da3_per_point(
    gaussians: Gaussians3D,
    reference_depth: np.ndarray,
    focal_x_px: float,
    focal_y_px: float,
    image_width: int,
    image_height: int,
) -> Gaussians3D:
    """Each Gaussian gets the median scale of its nearest DA3 Voronoi cell."""
    ctx = _voronoi_common(gaussians, reference_depth, focal_x_px, focal_y_px, image_width, image_height)
    if ctx is None:
        return gaussians

    valid_idx = ctx["valid_idx"]
    raw_scales = ctx["raw_scales"]
    within = ctx["within"]
    anchor_row = ctx["anchor_row"]
    anchor_col = ctx["anchor_col"]
    grid_w = ctx["grid_w"]
    global_median = ctx["global_median"]

    per_gauss_scale = np.full(len(ctx["pixel_x"]), global_median, dtype=np.float32)

    anchor_flat = anchor_row * grid_w + anchor_col
    sort_order = np.argsort(anchor_flat[within])
    sorted_anchors = anchor_flat[within][sort_order]
    sorted_scales = raw_scales[within][sort_order]
    boundaries = np.flatnonzero(np.diff(sorted_anchors)) + 1
    groups = np.split(sorted_scales, boundaries)
    unique_anchors = sorted_anchors[np.concatenate([[0], boundaries])]
    cell_medians = np.array([np.median(g) for g in groups], dtype=np.float32)
    print(f"  [DA3 per_point] Voronoi cells: {len(unique_anchors)}")

    anchor_to_median = dict(zip(unique_anchors.tolist(), cell_medians.tolist()))
    per_gauss_scale[valid_idx[within]] = np.array(
        [anchor_to_median[a] for a in anchor_flat[within].tolist()], dtype=np.float32
    )
    return _apply_per_gauss_scale(gaussians, per_gauss_scale)


def align_da3_zslab(
    gaussians: Gaussians3D,
    reference_depth: np.ndarray,
    focal_x_px: float,
    focal_y_px: float,
    image_width: int,
    image_height: int,
    num_slabs: int,
    max_depth: float,
) -> Gaussians3D:
    """Gaussians grouped into thin Z-depth bands; each band moves as a rigid unit."""
    ctx = _voronoi_common(gaussians, reference_depth, focal_x_px, focal_y_px, image_width, image_height)
    if ctx is None:
        return gaussians

    valid_idx = ctx["valid_idx"]
    valid_dz = ctx["valid_dz"]
    raw_scales = ctx["raw_scales"]
    within = ctx["within"]
    global_median = ctx["global_median"]

    slab_thickness = max_depth / num_slabs
    slab_idx = np.clip((valid_dz / slab_thickness).astype(np.int32), 0, num_slabs - 1)

    within_slab_idx = slab_idx[within]
    sort_order = np.argsort(within_slab_idx)
    sorted_slab = within_slab_idx[sort_order]
    sorted_scales_w = raw_scales[within][sort_order]

    boundaries = np.flatnonzero(np.diff(sorted_slab)) + 1
    slab_groups = np.split(np.arange(len(sorted_slab)), boundaries)
    unique_slabs = sorted_slab[np.concatenate([[0], boundaries])]
    slab_medians = np.array(
        [np.median(sorted_scales_w[g]) for g in slab_groups], dtype=np.float32
    )
    print(
        f"  [DA3 zslab] Occupied slabs: {len(unique_slabs)} / {num_slabs} "
        f"(thickness: {slab_thickness:.3f}m)"
    )

    per_gauss_scale = np.full(len(ctx["pixel_x"]), global_median, dtype=np.float32)
    slab_to_median = dict(zip(unique_slabs.tolist(), slab_medians.tolist()))
    for s, scale in slab_to_median.items():
        in_slab = valid_idx[slab_idx == s]
        per_gauss_scale[in_slab] = scale

    return _apply_per_gauss_scale(gaussians, per_gauss_scale)
