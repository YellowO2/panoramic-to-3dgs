import torch
import numpy as np
import cv2
from sharp.utils.gaussians import Gaussians3D
from depth_anything_3.api import DepthAnything3

def get_da3_predictions(image_paths, export_dir=None, model_type="./models/models--depth-anything--DA3NESTED-GIANT-LARGE-1.1/snapshots/b2359bdf726fb44ef62acca04d629dcf158053e7", device="cuda"):
    print(f"Loading Depth Anything 3 model '{model_type}' on {device}...")
    model = DepthAnything3.from_pretrained(model_type).to(device=device)
    
    print(f"Running DA3 inference on {len(image_paths)} views...")
    if export_dir:
        print(f"Exporting GLB to {export_dir}...")
        prediction = model.inference(image_paths, export_dir=export_dir, export_format="glb")
    else:
        prediction = model.inference(image_paths)
    return prediction


def scale_splat_to_depthmap(gaussians: Gaussians3D, ref_depth_map: np.ndarray, focal_px: float, width: int, height: int) -> Gaussians3D:
    # === 1. project 3dgs to depthmap ===
    positions = gaussians.mean_vectors[0] # Shape: (N, 3)
    z = positions[:, 2]
    valid = z > 0.1
    # convert to pixel coordinates
    px = (positions[:, 0] / z) * focal_px + (width / 2.0)
    py = (positions[:, 1] / z) * focal_px + (height / 2.0)
    valid = valid & (px >= 0) & (px < width - 1) & (py >= 0) & (py < height - 1)
    
    valid_z = z[valid].detach().cpu().numpy()
    valid_px = px[valid].long().detach().cpu().numpy()
    valid_py = py[valid].long().detach().cpu().numpy()
    
    # === 2. Obtain depthmap from DA3 ===
    da3_z = ref_depth_map[valid_py, valid_px]
    
    # === 3. Compute scale ratio ===
    ratios = da3_z / valid_z
    median_scale = float(np.median(ratios)) # we use median to reduce influence from outliers
    print(f"Calculated scale ratio: {median_scale:.4f}")
    
    # === 4. Apply the scale ===
    # scale both the position & size of the blobs
    scaled_mean_vectors = gaussians.mean_vectors * median_scale
    scaled_singular_values = gaussians.singular_values * median_scale
    return Gaussians3D(
        mean_vectors=scaled_mean_vectors,
        singular_values=scaled_singular_values,
        quaternions=gaussians.quaternions,
        colors=gaussians.colors,
        opacities=gaussians.opacities,
    )

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
  
    # Scale the positions and scales uniformly by the chosen factor.
    aligned_gaussians = scale_splat_to_depthmap(gaussians, reference_depth_view, focal_x_px, image_width, image_height)
    return aligned_gaussians

