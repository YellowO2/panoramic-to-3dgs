
import torch
import numpy as np
import math
from scipy.spatial.transform import Rotation
from sharp.utils.gaussians import Gaussians3D, apply_transform


def trim_splat_by_fov(gaussians: Gaussians3D, hfov_limit: float, vfov_limit: float = None) -> Gaussians3D:
    # this function trims away dirty edges of the splat for a clean cut.
    positions = gaussians.mean_vectors
    x= positions[0][:,0]
    z = positions[0][:, 2]
    half_hfov_rad = math.radians(hfov_limit / 2.0)
    max_x_ratio = math.tan(half_hfov_rad)
    mask = (z>0) & (torch.abs(x / z) <= max_x_ratio) # trim away points out of hfov specified

    #TODO: maybe implement for vfov, but not necessary now.
    return filter_gaussians(gaussians, mask)

def rotate_splat(gaussians: Gaussians3D, yaw: float, pitch: float) -> Gaussians3D:
    device = gaussians.mean_vectors.device # ensure rotation matrix is moved to the same gpu as the splat.
    rotation = Rotation.from_euler('yx', [yaw, pitch], degrees=True)
    # add 3x1 zero column to represent no translation.
    transform_with_translation = torch.cat([
        torch.tensor(rotation.as_matrix(), dtype=torch.float32).to(device), 
        torch.zeros((3, 1)).to(device) #since just a small matrix, we can just copy to the GPU without creating there.
    ], dim=1)
    return apply_transform(gaussians, transform_with_translation)

def merge_gaussians(splats_list: list[Gaussians3D]) -> Gaussians3D:
    # combine the splats into one
    return Gaussians3D(
        mean_vectors=torch.cat([item.mean_vectors for item in splats_list], dim=1),
        singular_values=torch.cat([item.singular_values for item in splats_list], dim=1),
        quaternions=torch.cat([item.quaternions for item in splats_list], dim=1),
        colors=torch.cat([item.colors for item in splats_list], dim=1),
        opacities=torch.cat([item.opacities for item in splats_list], dim=1),
    )

def filter_gaussians(gaussians: Gaussians3D, mask: torch.Tensor) -> Gaussians3D:
    return Gaussians3D(
        mean_vectors=gaussians.mean_vectors[:, mask, :],
        singular_values=gaussians.singular_values[:, mask, :],
        quaternions=gaussians.quaternions[:, mask, :],
        colors=gaussians.colors[:, mask, :],
        opacities=gaussians.opacities[:, mask],
    )

def align_splats_to_depthmap(splats_list: list[Gaussians3D], views: list) -> list[Gaussians3D]:
    from functions.depth_align import get_da3_predictions, align_gaussians_to_reference
    import torch
    import numpy as np
    
    # 1. Extract paths and run inference
    image_paths = [v["path"] for v in views]
    prediction = get_da3_predictions(image_paths)
    
    aligned_splats = []
    # prediction.depth shape: [N, H, W]
    for i, (view, splat, depth_map) in enumerate(zip(views, splats_list, prediction.depth)):
        focal_px = float(view["focal_px"])
        img_w = int(view["width"])
        img_h = int(view["height"])
        
        # 2. Align the current splat slice to the DA3 depth map
        aligned_gaussians, median_scale, count = align_gaussians_to_reference(
            gaussians=splat,
            reference_depth_view=depth_map,
            focal_x_px=focal_px,
            focal_y_px=focal_px,  # SHARP uses uniform focal length typically
            image_width=img_w,
            image_height=img_h,
            grid_resolution=8,
            detail_weight=0.0
        )
        print(f"Aligned splat {i}/{len(views)} with DA3 - items checked: {count}, median scale ratio: {median_scale:.4f}")
        aligned_splats.append(aligned_gaussians)

    return aligned_splats

def process_splats(views: list, splats_list: list[Gaussians3D], enable_alignment: bool = True) -> Gaussians3D:
    # main orchestrator for post-processing the generated 3dgs slices.
    
    if enable_alignment:
        print("Starting Depth Anything 3 multi-view alignment...")
        splats_list = align_splats_to_depthmap(splats_list, views)

    processed_splats = []
    # splats and their corresponding view data
    for view, splat_group in zip(views, splats_list):
        
        # 1. trim away the noise splats edges for a clean cut.
        hfov_keep = view["hfov"] - 8.0
        cleaned_splat = trim_splat_by_fov(splat_group, hfov_limit=hfov_keep)
        
        # 2. rotate the splats to where they are supposed to be.
        rotated_splat = rotate_splat(cleaned_splat, yaw=view["yaw"], pitch=view["pitch"])

        # TODO: Step 3. Translation scaling / alignment
        processed_splats.append(rotated_splat)
        
    return merge_gaussians(processed_splats)