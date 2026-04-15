
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
    device = gaussians.mean_vectors.device # ensure same gpu used.
    rotation = Rotation.from_euler('yx', [yaw, pitch], degrees=True)
    # add 3x1 zero column to represent no translation.
    transform_with_translation = torch.cat([
        torch.tensor(rotation.as_matrix(), dtype=torch.float32).to(device), 
        torch.zeros((3, 1)).to(device)
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

def process_splats(views: list, splats_list: list[Gaussians3D]) -> Gaussians3D:
    # main orchestrator for post-processing the generated 3dgs slices.

    processed_splats = []
    # splats and their corresponding view data
    for view, splat_group in zip(views, splats_list):
        
        # 1. trim away the noise splats edges for a clean cut.
        hfov_keep = view["hfov"] - 5.0
        cleaned_splat = trim_splat_by_fov(splat_group, hfov_limit=hfov_keep)
        
        # 2. rotate the splats to where they are supposed to be.
        rotated_splat = rotate_splat(cleaned_splat, yaw=view["yaw"], pitch=view["pitch"])

        # TODO: Step 3. Translation scaling / alignment
        processed_splats.append(rotated_splat)
        
    return merge_gaussians(processed_splats)