import torch
import numpy as np
from depth_anything_3.api import DepthAnything3
from datatype import View

class DA3Model:
    def __init__(self, model_path="./models/models--depth-anything--DA3NESTED-GIANT-LARGE-1.1/snapshots/b2359bdf726fb44ef62acca04d629dcf158053e7", device="cuda"):
        self.device = device
        print(f"Loading Depth Anything 3 model from '{model_path}' on {device}...")
        self.model = DepthAnything3.from_pretrained(model_path).to(device=device)

    def process_views(self, views: list[View], export_dir=None):
        """
        Runs multi-view inference and returns a map of pano_id -> median_center (3,)
        """
        if not views: return {}
        prediction = self.model.inference([v.path for v in views], export_dir=export_dir, export_format="glb" if export_dir else "mini_npz")
        
        # Store results in views
        for i, v in enumerate(views):
            v.depth = prediction.depth[i]
            v.extrinsics = prediction.extrinsics[i]
            
        # Calculate shared centers per pano_id
        pano_centers = {}
        pano_groups = {}
        for v in views:
            if v.pano_id not in pano_groups: pano_groups[v.pano_id] = []
            pano_groups[v.pano_id].append(v)
            
        for pano_id, group in pano_groups.items():
            # Get camera positions (C = -R^T @ t)
            centers = []
            for v in group:
                R, t = v.extrinsics[:, :3], v.extrinsics[:, 3:]
                centers.append((-R.T @ t).flatten())
            
            median_center = np.median(centers, axis=0)
            pano_centers[pano_id] = median_center
            
            # Snap horizon slices to this center
            for v in group:
                R = v.extrinsics[:, :3]
                v.extrinsics[:, 3] = (-R @ median_center.reshape(3, 1)).flatten()
                
        return pano_centers
