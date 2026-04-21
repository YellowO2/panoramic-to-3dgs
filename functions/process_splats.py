
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

def align_splats_to_depthmap(splats_list: list[Gaussians3D], views: list) -> tuple:
    from functions.depth_align import get_da3_predictions, align_gaussians_to_reference
    import torch
    import numpy as np
    import os
    
    # 0. Check if we already have pre-computed DA360 depth maps in the views
    use_da360 = all("da360_depth" in v for v in views)
    
    # 1. Extract paths and run inference if not using DA360
    if not use_da360:
        image_paths = [v["path"] for v in views]
        debug_dir = "da3_debug_output"
        os.makedirs(debug_dir, exist_ok=True)
        prediction = get_da3_predictions(image_paths, export_dir=debug_dir)
        depth_maps = prediction.depth
        extrinsics = prediction.extrinsics
    else:
        print("Using DA360 provided depth maps for alignment.")
        depth_maps = [v["da360_depth"] for v in views]
        extrinsics = None # DA360 just does depth, not pose
    
    aligned_splats = []
    # prediction.depth shape: [N, H, W]
    for i, (view, splat, depth_map) in enumerate(zip(views, splats_list, depth_maps)):
        focal_px = float(view["focal_px"])
        img_w = int(view["width"])
        img_h = int(view["height"])
        
        # 2. Align the current splat slice to the depth map
        aligned_gaussians = align_gaussians_to_reference(
            gaussians=splat,
            reference_depth_view=depth_map,
            focal_x_px=focal_px,
            focal_y_px=focal_px,  # SHARP uses uniform focal length typically
            image_width=img_w,
            image_height=img_h,
            grid_resolution=8,
            detail_weight=0.0
        )
        print(f"Aligned splat {i}/{len(views)} with Depth Map")
        aligned_splats.append(aligned_gaussians)

    return aligned_splats, extrinsics

def process_splats(views: list, splats_list: list[Gaussians3D], enable_alignment: bool = True) -> Gaussians3D:
    # main orchestrator for post-processing the generated 3dgs slices.
    
    extrinsics = None
    if enable_alignment:
        print("Starting Depth Anything 3 multi-view alignment...")
        splats_list, extrinsics = align_splats_to_depthmap(splats_list, views)

    processed_splats = []
    # splats and their corresponding view data
    for i, (view, splat_group) in enumerate(zip(views, splats_list)):
        
        # 1. trim away the noise splats edges for a clean cut.
        hfov_keep = view["hfov"] - 6.0
        cleaned_splat = trim_splat_by_fov(splat_group, hfov_limit=hfov_keep)
        
        # 2. Translate and scaling / alignment
        if enable_alignment and extrinsics is not None:
            # extrinsics is a stack of 3x4 W2C (World to Camera) matrices
            w2c = extrinsics[i] # [3, 4] numpy array
            
            # We want to transform the splat from local camera space to a global world space
            # So we need C2W (Camera to World) matrix, which is the inverse of W2C.
            r_w2c = w2c[:, :3]
            t_w2c = w2c[:, 3:]
            
            # Inverse of a rotation matrix is its transpose. Translation is -R^T * t
            r_c2w = r_w2c.T
            t_c2w = -r_c2w @ t_w2c
            c2w = np.hstack((r_c2w, t_c2w))
            
            # Apply transformation
            device = cleaned_splat.mean_vectors.device
            c2w_tensor = torch.tensor(c2w, dtype=torch.float32, device=device)
            rotated_splat = apply_transform(cleaned_splat, c2w_tensor)
        else:
            # 2. rotate the splats to where they are supposed to be manually (fallback).
            rotated_splat = rotate_splat(cleaned_splat, yaw=view["yaw"], pitch=view["pitch"])

        processed_splats.append(rotated_splat)
        
    return merge_gaussians(processed_splats)