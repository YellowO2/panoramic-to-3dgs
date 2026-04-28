import torch
import numpy as np
from scipy.spatial.transform import Rotation
from depth_anything_3.api import DepthAnything3
from datatype import View

class DA3Result:
    def __init__(self, pano_poses, prediction):
        self.pano_poses = pano_poses # pano_id -> {center, rotation}
        self.prediction = prediction # Filtered DA3 Prediction object

class DA3Model:
    def __init__(self, model_path="./models/models--depth-anything--DA3NESTED-GIANT-LARGE-1.1/snapshots/b2359bdf726fb44ef62acca04d629dcf158053e7", device="cuda"):
        self.device = device
        print(f"Loading Depth Anything 3 model from '{model_path}' on {device}...")
        self.model = DepthAnything3.from_pretrained(model_path).to(device=device)

    def process_views(self, views: list[View], dist_thresh=0.08, angle_thresh=5.0):
        """
        Runs multi-view inference, filters out views that deviate from expected
        Shared Center and Yaw/Pitch values, and returns the cleaned result.
        """
        if not views: return [], DA3Result({}, None)
        
        prediction = self.model.inference([v.path for v in views], export_format="mini_npz")
        
        # 1. Group indices by panorama
        pano_groups = {}
        for i, v in enumerate(views):
            if v.pano_id not in pano_groups: pano_groups[v.pano_id] = []
            pano_groups[v.pano_id].append(i)
            
        keep_indices = []
        final_pano_poses = {}

        for pano_id, indices in pano_groups.items():
            # A. Calculate "Consensus" Pose for this Panorama
            centers = []
            global_rots = [] # Store predicted R_pano candidates
            
            for idx in indices:
                v = views[idx]
                w2c = prediction.extrinsics[idx]
                R_w2c, t_w2c = w2c[:3, :3], w2c[:3, 3:]
                centers.append((-R_w2c.T @ t_w2c).flatten())
                
                # R_w2c = R_local.T @ R_pano  => R_pano = R_local @ R_w2c
                R_local = Rotation.from_euler('yx', [v.yaw, v.pitch], degrees=True).as_matrix()
                global_rots.append(R_local @ R_w2c)
            
            median_center = np.median(centers, axis=0)
            # Average rotations via quaternion mean
            quats = np.array([Rotation.from_matrix(R).as_quat() for R in global_rots])
            quats *= np.sign(quats @ quats[0])  # flip to same hemisphere
            consensus_pano_rot = Rotation.from_quat(quats.mean(axis=0)).as_matrix()
            
            # B. Filter Outliers
            pano_keep = []
            for i, idx in enumerate(indices):
                v = views[idx]
                
                # 1. Translation Deviation
                dist = np.linalg.norm(centers[i] - median_center)
                
                # 2. Rotation Deviation
                # Calculate expected W2C rotation: R_expected = R_local.T @ R_pano_consensus
                R_local = Rotation.from_euler('yx', [v.yaw, v.pitch], degrees=True).as_matrix()
                R_expected = R_local.T @ consensus_pano_rot
                R_pred = prediction.extrinsics[idx][:3, :3]
                
                # Error rotation
                R_err = R_pred @ R_expected.T
                # Convert to angle
                angle_err = np.degrees(np.arccos(np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)))
                
                if dist <= dist_thresh and angle_err <= angle_thresh:
                    pano_keep.append(idx)
                else:
                    print(f"Filtering view {v.path}: dev_dist={dist:.3f}m, dev_angle={angle_err:.1f}deg")
            
            if pano_keep:
                keep_indices.extend(pano_keep)
                final_pano_poses[pano_id] = {'center': median_center, 'rotation': consensus_pano_rot}
                # Force strictly shared center for the kept ones
                for idx in pano_keep:
                    R = prediction.extrinsics[idx][:3, :3]
                    prediction.extrinsics[idx, :3, 3] = (-R @ median_center.reshape(3, 1)).flatten()

        # 3. Create filtered result
        keep_indices = sorted(keep_indices)
        filtered_views = [views[i] for i in keep_indices]
        
        # Filter Prediction arrays
        prediction.depth = prediction.depth[keep_indices]
        prediction.extrinsics = prediction.extrinsics[keep_indices]
        prediction.intrinsics = prediction.intrinsics[keep_indices]
        if prediction.conf is not None:
            prediction.conf = prediction.conf[keep_indices]
        if hasattr(prediction, 'processed_images') and prediction.processed_images is not None:
            prediction.processed_images = [prediction.processed_images[i] for i in keep_indices]
            
        print(f"Cleaned scene: Kept {len(filtered_views)}/{len(views)} views.")
        return filtered_views, DA3Result(final_pano_poses, prediction)
