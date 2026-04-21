import torch
import numpy as np
import cv2
from sharp.utils.gaussians import Gaussians3D
from depth_anything_3.api import DepthAnything3

def get_da3_predictions(image_paths, model_type="./models/models--depth-anything--DA3NESTED-GIANT-LARGE-1.1/snapshots/b2359bdf726fb44ef62acca04d629dcf158053e7", device="cuda"):
    """
    Runs the Depth Anything 3 API to produce globally scaled depth maps.
    Returns the prediction object, containing prediction.depth of shape [N, H, W]
    """
    print(f"Loading Depth Anything 3 model '{model_type}' on {device}...")
    model = DepthAnything3.from_pretrained(model_type)
    model = model.to(device=device)
    
    print(f"Running DA3 inference on {len(image_paths)} views...")
    prediction = model.inference(image_paths)
    return prediction

def bilinear_sample_scalar(image: np.ndarray, sample_x: np.ndarray, sample_y: np.ndarray) -> np.ndarray:
    """
    Samples scalar values from a 2D image array using bilinear interpolation.
    Clamps edges for normal perspective fields.
    """
    height, width = image.shape
    x0 = np.clip(np.floor(sample_x).astype(np.int32), 0, width - 1)
    y0 = np.clip(np.floor(sample_y).astype(np.int32), 0, height - 1)
    x1 = np.clip(x0 + 1, 0, width - 1)
    y1 = np.clip(y0 + 1, 0, height - 1)

    wx = sample_x - x0
    wy = sample_y - y0

    image_f32 = image.astype(np.float32)
    s00 = image_f32[y0, x0]
    s01 = image_f32[y0, x1]
    s10 = image_f32[y1, x0]
    s11 = image_f32[y1, x1]

    # Bilinear mix
    return (s00 * (1 - wx) * (1 - wy) +
            s01 * wx * (1 - wy) +
            s10 * (1 - wx) * wy +
            s11 * wx * wy)

def align_gaussians_to_reference(
    gaussians: Gaussians3D,
    reference_depth_view: np.ndarray,
    focal_x_px: float,
    focal_y_px: float,
    image_width: int,
    image_height: int,
    grid_resolution: int = 8,
    detail_weight: float = 0.0,
) -> tuple[Gaussians3D, float, int]:
    """
    Align Gaussian depths to a reference Depth Anything map using a smooth low-frequency scale field.
    This preserves the internal structural layout of the 3D objects to avoid flattening.
    """
    grid_cells_x = max(1, int(grid_resolution))
    grid_cells_y = max(1, int(round(grid_cells_x * (image_height / max(1, image_width)))))

    mean_vectors = gaussians.mean_vectors  # (1, N, 3)
    mv_np = mean_vectors[0].detach().cpu().numpy().astype(np.float32)
    depth_z = mv_np[:, 2]
    radial = np.linalg.norm(mv_np, axis=1)

    valid = depth_z > 1e-6
    pixel_x = (mv_np[:, 0] / np.clip(depth_z, 1e-6, None)) * focal_x_px + (image_width / 2.0) - 0.5
    pixel_y = (mv_np[:, 1] / np.clip(depth_z, 1e-6, None)) * focal_y_px + (image_height / 2.0) - 0.5
    valid &= (pixel_x >= 0) & (pixel_x <= image_width - 1)
    valid &= (pixel_y >= 0) & (pixel_y <= image_height - 1)

    per_point_scale = np.ones(mv_np.shape[0], dtype=np.float32)
    median_scale = 1.0
    count = 0

    if int(valid.sum()) >= 64:
        ref_depth = bilinear_sample_scalar(
            reference_depth_view, pixel_x[valid], pixel_y[valid],
        )
        # Note: DA3 prediction.depth is metric relative depth, unlike raw disparity
        ok = np.isfinite(ref_depth) & (ref_depth > 1e-6) & (radial[valid] > 1e-6)
        count = int(ok.sum())
        if count >= 64:
            ref_depth_ok = ref_depth[ok] 
            sharp_r_ok = radial[valid][ok]
            
            # To pull Gaussian to reference distance: scale = DA3 Depth / SHARP distance
            raw_scale = ref_depth_ok / sharp_r_ok

            lo, hi = np.quantile(raw_scale, [0.05, 0.95])
            trimmed = raw_scale[(raw_scale >= lo) & (raw_scale <= hi)]
            median_scale = float(np.median(trimmed)) if trimmed.size > 0 else float(np.median(raw_scale))

            px_ok = pixel_x[valid][ok]
            py_ok = pixel_y[valid][ok]

            cell_width = image_width / grid_cells_x
            cell_height = image_height / grid_cells_y
            grid = np.full((grid_cells_y, grid_cells_x), median_scale, dtype=np.float32)
            
            for gy in range(grid_cells_y):
                for gx in range(grid_cells_x):
                    in_cell = (
                        (px_ok >= gx * cell_width) & (px_ok < (gx + 1) * cell_width)
                        & (py_ok >= gy * cell_height) & (py_ok < (gy + 1) * cell_height)
                    )
                    if int(in_cell.sum()) >= 8:
                        cell_scales = raw_scale[in_cell]
                        cl, ch = np.quantile(cell_scales, [0.1, 0.9])
                        cell_trimmed = cell_scales[(cell_scales >= cl) & (cell_scales <= ch)]
                        if cell_trimmed.size > 0:
                            grid[gy, gx] = float(np.median(cell_trimmed))

            grid = np.clip(grid, median_scale * 0.1, median_scale * 10.0)

            all_px = np.clip(pixel_x, 0, image_width - 1)
            all_py = np.clip(pixel_y, 0, image_height - 1)
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
            smooth_scale = (
                s00 * (1 - wx) * (1 - wy)
                + s01 * wx * (1 - wy)
                + s10 * (1 - wx) * wy
                + s11 * wx * wy
            )

            dw = float(np.clip(detail_weight, 0.0, 1.0))
            if dw > 0.0:
                per_point_raw = np.full(mv_np.shape[0], median_scale, dtype=np.float32)
                valid_indices = np.where(valid)[0]
                ok_within_valid = np.where(ok)[0]
                per_point_raw[valid_indices[ok_within_valid]] = raw_scale
                per_point_scale = smooth_scale * (1.0 - dw) + per_point_raw * dw
            else:
                per_point_scale = smooth_scale
            
            per_point_scale[~valid] = median_scale

    device = mean_vectors.device
    dtype = mean_vectors.dtype
    scale_t = (
        torch.from_numpy(per_point_scale)
        .to(device=device, dtype=dtype)
        .unsqueeze(0)
        .unsqueeze(-1)
    )  # (1, N, 1)

    # Scale the positions and scales by the chosen factor.
    aligned_gaussians = Gaussians3D(
        mean_vectors=mean_vectors * scale_t,
        singular_values=gaussians.singular_values * scale_t,
        quaternions=gaussians.quaternions,
        colors=gaussians.colors,
        opacities=gaussians.opacities,
    )
    return aligned_gaussians, median_scale, count
