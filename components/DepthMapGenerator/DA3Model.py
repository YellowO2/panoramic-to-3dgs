import torch
import numpy as np
from scipy.spatial.transform import Rotation
from depth_anything_3.api import DepthAnything3
from datatype import View

class DA3Result:
    def __init__(self, pano_poses, prediction):
        self.pano_poses = pano_poses # pano_id -> {center, rotation}
        self.prediction = prediction # Full DA3 Prediction object

class DA3Model:
    def __init__(self, model_path="./models/models--depth-anything--DA3NESTED-GIANT-LARGE-1.1/snapshots/b2359bdf726fb44ef62acca04d629dcf158053e7", device="cuda"):
        self.device = device
        print(f"Loading Depth Anything 3 model from '{model_path}' on {device}...")
        self.model = DepthAnything3.from_pretrained(model_path).to(device=device)

    def process_views(self, views: list[View], export_dir=None):
        if not views: return DA3Result({}, None)
        
        prediction = self.model.inference([v.path for v in views], export_dir=export_dir, export_format="mini_npz")
        
        # 1. Update View depths
        for i, v in enumerate(views):
            v.depth = prediction.depth[i]
            
        # 2. Extract shared poses
        pano_poses = {}
        pano_groups = {}
        for i, v in enumerate(views):
            if v.pano_id not in pano_groups: pano_groups[v.pano_id] = []
            pano_groups[v.pano_id].append((v, i))
            
        for pano_id, group in pano_groups.items():
            centers = []
            global_rots = []
            for v, idx in group:
                w2c = prediction.extrinsics[idx]
                R_w2c, t_w2c = w2c[:3, :3], w2c[:3, 3:]
                centers.append((-R_w2c.T @ t_w2c).flatten())
                R_local = Rotation.from_euler('yx', [v.yaw, v.pitch], degrees=True).as_matrix()
                global_rots.append(R_local.T @ R_w2c)
            
            median_center = np.median(centers, axis=0)
            median_rot = global_rots[0]
            pano_poses[pano_id] = {'center': median_center, 'rotation': median_rot}
            
            # Snap extrinsics in the Prediction object to the shared center
            for v, idx in group:
                w2c = prediction.extrinsics[idx]
                R = w2c[:3, :3]
                prediction.extrinsics[idx, :3, 3] = (-R @ median_center.reshape(3, 1)).flatten()
        
        return DA3Result(pano_poses, prediction)
