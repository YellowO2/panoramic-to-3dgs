import copy
import numpy as np
from scipy.spatial.transform import Rotation
from depth_anything_3.api import DepthAnything3
from datatype import View

class DA3Result:
    def __init__(self, pano_poses, prediction, pano_keep_counts=None, pano_avg_deviation=None):
        self.pano_poses = pano_poses # pano_id -> {center, rotation}
        self.prediction = prediction # Filtered DA3 Prediction object
        self.pano_avg_deviation = pano_avg_deviation or {} # pano_id -> avg dist deviation (m) among its own KEPT views only -- a single wild outlier gets filtered out anyway, so it says nothing about quality; the kept views still not agreeing well with each other on average is what actually flags a bad pairing. inf if zero views were kept.
        self.pano_keep_counts = pano_keep_counts or {} # pano_id -> (kept, total)

class DA3Model:
    def __init__(self, model_path="./models/models--depth-anything--DA3NESTED-GIANT-LARGE-1.1/snapshots/b2359bdf726fb44ef62acca04d629dcf158053e7", device="cuda"):
        self.device = device
        print(f"Loading Depth Anything 3 model from '{model_path}' on {device}...")
        self.model = DepthAnything3.from_pretrained(model_path).to(device=device)

    def _consensus_pose(self, indices: list[int], views: list[View], prediction):
        """Median center + mean-quaternion rotation across exactly the given
        view indices. Shared by the first pass (all slices of a pano, in
        _compute_pano_consensus) and the second pass (just the slices that
        survived the deviation threshold, in _filter_at_threshold) -- the
        second pass matters because a plain quaternion mean isn't robust to
        outliers the way the median center is, so a pano's rotation can stay
        skewed by the very slices about to be dropped unless it's
        recomputed from the survivors alone."""
        centers = {}
        global_rots = {}
        R_locals = {}
        for idx in indices:
            v = views[idx]
            w2c = prediction.extrinsics[idx]
            R_w2c, t_w2c = w2c[:3, :3], w2c[:3, 3:]
            centers[idx] = (-R_w2c.T @ t_w2c).flatten()

            # R_w2c = R_local.T @ R_pano  => R_pano = R_local @ R_w2c
            R_local = Rotation.from_euler('yx', [v.yaw, v.pitch], degrees=True).as_matrix()
            R_locals[idx] = R_local
            global_rots[idx] = R_local @ R_w2c

        median_center = np.median(list(centers.values()), axis=0)
        quats = np.array([Rotation.from_matrix(global_rots[idx]).as_quat() for idx in indices])
        quats *= np.sign(quats @ quats[0])[:, None]  # flip to same hemisphere
        consensus_rot = Rotation.from_quat(quats.mean(axis=0)).as_matrix()
        return median_center, consensus_rot, centers, R_locals

    def _compute_pano_consensus(self, views: list[View], prediction):
        """Per-pano consensus pose (median center + mean-quaternion rotation,
        from ALL of a pano's slices) plus each view's own (dist, angle_err)
        deviation from it, and the R_local used to compute that deviation --
        all threshold-independent, so this only needs to run once per
        inference call regardless of how many (dist_thresh, angle_thresh)
        levels get applied afterward. This first-pass consensus is only used
        to decide which slices deviate too far to keep -- the final pose
        used downstream is recomputed from just the survivors, see
        _filter_at_threshold."""
        pano_groups = {}
        for i, v in enumerate(views):
            pano_groups.setdefault(v.pano_id, []).append(i)

        per_pano = {}
        for pano_id, indices in pano_groups.items():
            median_center, consensus_pano_rot, centers, R_locals = self._consensus_pose(indices, views, prediction)

            per_view = []
            for idx in indices:
                R_local = R_locals[idx]
                dist = np.linalg.norm(centers[idx] - median_center)

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
        against the same inference result (see process_views_sweep).

        Pose availability is NOT gated on filtering: filtered_views/
        keep_indices (which slices actually contribute points downstream)
        and final_pano_poses (every pano's own position estimate) are
        separate concerns -- a pano gets a real pose in final_pano_poses
        regardless of how many (if any) of its views survive the per-view
        cutoff, falling back to the raw pre-filter consensus when nothing
        does. Making pose depend on filtering succeeding was the wrong
        coupling: a caller doing its OWN quality judgment on top (e.g. a
        client-side bridging search comparing several candidate pairs)
        needs the actual number to reason about, not silence."""
        keep_indices = []
        final_pano_poses = {}
        pano_keep_counts = {}
        pano_avg_deviation = {}
        snapped_extrinsics = prediction.extrinsics.copy()

        for pano_id, data in per_pano.items():
            pano_keep = []
            for pv in data['per_view']:
                if pv['dist'] <= dist_thresh and pv['angle_err'] <= angle_thresh:
                    pano_keep.append(pv['idx'])
                else:
                    v = views[pv['idx']]
                    print(f"Filtering view {v.path}: dev_dist={pv['dist']:.3f}m, dev_angle={pv['angle_err']:.1f}deg")

            # Average deviation among the SURVIVING views only -- not the
            # worst outlier across everyone. A single wild outlier gets
            # filtered out by the check above anyway (that's the whole
            # point of it), so it says nothing about the pairing's real
            # quality. What actually indicates a bad pairing is the kept
            # views still not agreeing well with each other on average.
            kept_devs = [pv['dist'] for pv in data['per_view'] if pv['idx'] in pano_keep]
            pano_avg_deviation[pano_id] = sum(kept_devs) / len(kept_devs) if kept_devs else float('inf')

            if pano_keep:
                keep_indices.extend(pano_keep)
                # Recompute consensus from just the surviving slices -- the
                # first pass's consensus (data['center']/['rotation']) still
                # included the outliers being dropped here, which can skew
                # it (the quaternion-mean rotation especially, see
                # _consensus_pose). This is the pose actually used downstream.
                final_center, final_rot, _, R_locals = self._consensus_pose(pano_keep, views, prediction)
                final_pano_poses[pano_id] = {'center': final_center, 'rotation': final_rot}
                # Snap kept views to the recomputed consensus pose.
                for pv in data['per_view']:
                    if pv['idx'] not in pano_keep:
                        continue
                    R_snapped = R_locals[pv['idx']].T @ final_rot
                    snapped_extrinsics[pv['idx'], :3, :3] = R_snapped
                    snapped_extrinsics[pv['idx'], :3, 3] = (-R_snapped @ final_center.reshape(3, 1)).flatten()
                cx, cy, cz = final_center
            else:
                # Nothing survived the per-view threshold -- pose
                # availability shouldn't depend on filtering succeeding
                # (that's a separate concern, see this method's docstring
                # note below). Fall back to the raw, threshold-independent
                # first-pass consensus computed in _compute_pano_consensus
                # from ALL of this pano's own views -- worse than a
                # filtered consensus, but still a real multi-view estimate,
                # not nothing.
                final_pano_poses[pano_id] = {'center': data['center'], 'rotation': data['rotation']}
                cx, cy, cz = data['center']

            pano_keep_counts[pano_id] = (len(pano_keep), len(data['per_view']))
            print(f"  pano {pano_id}: kept {len(pano_keep)}/{len(data['per_view'])}, center=({cx:.3f}, {cy:.3f}, {cz:.3f})")

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
        return filtered_views, DA3Result(final_pano_poses, filtered_pred, pano_keep_counts, pano_avg_deviation)

    def process_views(self, views: list[View], dist_thresh=0.2, angle_thresh=1):
        """
        Runs multi-view inference, filters out views that deviate from expected
        shared center and yaw/pitch values, and returns the cleaned result.
        """
        return self.process_views_sweep(views, [(dist_thresh, angle_thresh)])[0]

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
