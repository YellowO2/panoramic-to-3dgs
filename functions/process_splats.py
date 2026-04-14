
import torch
import numpy as np


def trim_splat_by_fov(gaussians: Gaussians3D, hfov_limit: float, vfov_limit: float = None) -> Gaussians3D:
    """
    Takes a single Gaussians3D object and removes points outside the FOV limits.
    """
    # 1. Start here! Extract positions
    positions = gaussians.mean_vectors
    
    # 2. Get the Z depth (index 2)
    
    # 3. Create your first boolean mask (e.g. z > 0)
    
    # 4. Do the trig for hfov_limit to find the max X/Z ratio and update the mask
    
    # 5. Return the new filtered Gaussians3D object
    pass 


def process_splats(views: list, splats_list: list):
    """
    Main orchestrator for post-processing multiple views of 3D Gaussians.
    """
    processed_splats = []
    
    # Zipping views and their corresponding generated splats
    for view, splat_group in zip(views, splats_list):
        
        # 1. Trimming (Remove edges)
        # Using a buffer: if image was 90 deg, we keep 85 deg to remove the 2.5 deg edges
        hfov_keep = view["hfov"] - 5.0
        
        # You can also compute and pass vfov_keep similarly if needed.
        # Apple MLSharp tends to lose coherence past 80-90 degrees radially.
        cleaned_splat = trim_splats(splat_group, hfov_limit=hfov_keep)
        
        # TODO: Step 2. Rotations (Yaw/Pitch back to world space)
        # TODO: Step 3. Translation scaling / alignment

        processed_splats.append(cleaned_splat)
        
    return processed_splats