import torch
import numpy as np
import cv2
from sharp.utils.gaussians import Gaussians3D
from depth_anything_3.api import DepthAnything3

def get_da3_predictions(image_paths, model_type="./models/models--depth-anything--DA3NESTED-GIANT-LARGE-1.1/snapshots/b2359bdf726fb44ef62acca04d629dcf158053e7", device="cuda"):
    print(f"Loading Depth Anything 3 model '{model_type}' on {device}...")
    model = DepthAnything3.from_pretrained(model_type)
    model = model.to(device=device)
    
    print(f"Running DA3 inference on {len(image_paths)} views...")
    prediction = model.inference(image_paths)
    return prediction

def save_depth_to_ply(depth_map, image_path, focal_px, output_path):
    import cv2
    import numpy as np
    
    # Load color image
    color = cv2.imread(image_path)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    h, w = depth_map.shape
    
    # Create pixel grid
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    
    # Back-project to 3D
    z = depth_map
    x3d = (x - w/2) * z / focal_px
    y3d = (y - h/2) * z / focal_px
    
    # Stack points
    points = np.stack((x3d, y3d, z), axis=-1).reshape(-1, 3)
    colors = color.reshape(-1, 3)
    
    # Filter out zero depth
    valid = z.flatten() > 0
    points = points[valid]
    colors = colors[valid]
    
    # Save simple PLY format manually
    with open(output_path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, colors):
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {c[0]} {c[1]} {c[2]}\n")

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

    device = mean_vectors.device
    dtype = mean_vectors.dtype
    
    # FOR NOW (DEBUGGING): Just use the global median scale uniformly to prevent distortion!
    scale_t = torch.tensor([[[median_scale]]], device=device, dtype=dtype)

    # Scale the positions and scales uniformly by the chosen factor.
    aligned_gaussians = Gaussians3D(
        mean_vectors=mean_vectors * scale_t,
        singular_values=gaussians.singular_values * scale_t,
        quaternions=gaussians.quaternions,
        colors=gaussians.colors,
        opacities=gaussians.opacities,
    )
    return aligned_gaussians, median_scale, count

