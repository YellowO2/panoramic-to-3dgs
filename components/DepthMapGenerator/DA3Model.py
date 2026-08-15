import copy
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

    def _compute_pano_consensus(self, views: list[View], prediction):
        """Per-pano consensus pose (median center + mean-quaternion rotation)
        plus each view's own (dist, angle_err) deviation from it, and the
        R_local used to compute that deviation -- all threshold-independent,
        so this only needs to run once per inference call regardless of how
        many (dist_thresh, angle_thresh) levels get applied afterward."""
        pano_groups = {}
        for i, v in enumerate(views):
            pano_groups.setdefault(v.pano_id, []).append(i)

        per_pano = {}
        for pano_id, indices in pano_groups.items():
            centers = []
            global_rots = []
            R_locals = []
            for idx in indices:
                v = views[idx]
                w2c = prediction.extrinsics[idx]
                R_w2c, t_w2c = w2c[:3, :3], w2c[:3, 3:]
                centers.append((-R_w2c.T @ t_w2c).flatten())

                # R_w2c = R_local.T @ R_pano  => R_pano = R_local @ R_w2c
                R_local = Rotation.from_euler('yx', [v.yaw, v.pitch], degrees=True).as_matrix()
                R_locals.append(R_local)
                global_rots.append(R_local @ R_w2c)

            median_center = np.median(centers, axis=0)
            quats = np.array([Rotation.from_matrix(R).as_quat() for R in global_rots])
            quats *= np.sign(quats @ quats[0])[:, None]  # flip to same hemisphere
            consensus_pano_rot = Rotation.from_quat(quats.mean(axis=0)).as_matrix()

            per_view = []
            for i, idx in enumerate(indices):
                R_local = R_locals[i]
                dist = np.linalg.norm(centers[i] - median_center)

                R_expected = R_local.T @ consensus_pano_rot
                R_pred = prediction.extrinsics[idx][:3, :3]
                R_err = R_pred @ R_expected.T
                angle_err = np.degrees(np.arccos(np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)))

                per_view.append({'idx': idx, 'dist': dist, 'angle_err': angle_err, 'R_local': R_local})

            per_pano[pano_id] = {
                'center': median_center,
                'rotation': consensus_pano_rot,
                'per_view': per_view,
            }
        return per_pano

    def _filter_at_threshold(self, views: list[View], prediction, per_pano: dict, dist_thresh: float, angle_thresh: float):
        """Apply one (dist_thresh, angle_thresh) cutoff to already-computed
        per-view deviations, snapping kept views to their pano's consensus
        pose. Builds its own copy of the prediction arrays rather than
        mutating the shared one, so this is safe to call multiple times
        against the same inference result (see process_views_sweep)."""
        keep_indices = []
        final_pano_poses = {}
        snapped_extrinsics = prediction.extrinsics.copy()

        for pano_id, data in per_pano.items():
            pano_keep = []
            for pv in data['per_view']:
                if pv['dist'] <= dist_thresh and pv['angle_err'] <= angle_thresh:
                    pano_keep.append(pv['idx'])
                else:
                    v = views[pv['idx']]
                    print(f"Filtering view {v.path}: dev_dist={pv['dist']:.3f}m, dev_angle={pv['angle_err']:.1f}deg")

            if pano_keep:
                keep_indices.extend(pano_keep)
                final_pano_poses[pano_id] = {'center': data['center'], 'rotation': data['rotation']}
                # Snap kept views to consensus pose (shared center + consistent rotation)
                for pv in data['per_view']:
                    if pv['idx'] not in pano_keep:
                        continue
                    R_snapped = pv['R_local'].T @ data['rotation']
                    snapped_extrinsics[pv['idx'], :3, :3] = R_snapped
                    snapped_extrinsics[pv['idx'], :3, 3] = (-R_snapped @ data['center'].reshape(3, 1)).flatten()

        keep_indices = sorted(keep_indices)
        filtered_views = [views[i] for i in keep_indices]

        filtered_pred = copy.copy(prediction)
        filtered_pred.extrinsics = snapped_extrinsics[keep_indices]
        filtered_pred.depth = prediction.depth[keep_indices]
        filtered_pred.intrinsics = prediction.intrinsics[keep_indices]
        if prediction.conf is not None:
            filtered_pred.conf = prediction.conf[keep_indices]
        if hasattr(prediction, 'processed_images') and prediction.processed_images is not None:
            filtered_pred.processed_images = [prediction.processed_images[i] for i in keep_indices]

        print(f"Cleaned scene: Kept {len(filtered_views)}/{len(views)} views. (dist_thresh={dist_thresh}, angle_thresh={angle_thresh})")
        return filtered_views, DA3Result(final_pano_poses, filtered_pred)

    def process_views(self, views: list[View], dist_thresh=0.2, angle_thresh=1):
        """
        Runs multi-view inference, filters out views that deviate from expected
        shared center and yaw/pitch values, and returns the cleaned result.
        """
        if not views: return [], DA3Result({}, None)

        prediction = self.model.inference(
            [v.path for v in views],
            export_format="mini_npz",
        )
        per_pano = self._compute_pano_consensus(views, prediction)
        return self._filter_at_threshold(views, prediction, per_pano, dist_thresh, angle_thresh)

    def process_views_sweep(self, views: list[View], threshold_levels: list[tuple[float, float]]):
        """Debug/diagnostic helper: runs the multi-view inference forward pass
        exactly once, then applies each (dist_thresh, angle_thresh) pair in
        threshold_levels to that same result -- for comparing how much the
        consensus filter actually matters without paying for a separate GPU
        forward pass per level. Returns one (filtered_views, DA3Result) pair
        per threshold level, in the same order as threshold_levels."""
        if not views:
            return [([], DA3Result({}, None)) for _ in threshold_levels]

        prediction = self.model.inference(
            [v.path for v in views],
            export_format="mini_npz",
        )
        per_pano = self._compute_pano_consensus(views, prediction)
        return [
            self._filter_at_threshold(views, prediction, per_pano, dist_thresh, angle_thresh)
            for dist_thresh, angle_thresh in threshold_levels
        ]
