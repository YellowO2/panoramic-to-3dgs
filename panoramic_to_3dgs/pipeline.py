import os
import json
import tempfile
import contextlib
import numpy as np
import torch

from components.SplatGenerator.SplatGenerator import SplatGenerator
from components.DepthMapGenerator.DA3Model import DA3Model
from components.SplatProcessor.SplatProcessor import SplatProcessor
from components.ViewExtractor.ViewExtractor import extract_views, extract_views_for_da3
from components.Saver.Saver import Saver
from components.SplatProcessor.utils import backproject_views_to_pcd
from sharp.utils.gaussians import Gaussians3D, save_ply

from panoramic_to_3dgs.config import PipelineConfig


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in meters. Used by run_windowed_reconstruction
    (forced-overlap selection by distance to a window's boundary node) and
    run_windowed_reconstruction_full_pool (not directly, but kept alongside
    _rigid_align since both windowed-reconstruction variants need it)."""
    from math import atan2, cos, radians, sin, sqrt

    earth_radius_m = 6371000.0
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * earth_radius_m * atan2(sqrt(a), sqrt(1 - a))


def _rigid_align(shared_from: list[tuple[np.ndarray, np.ndarray]], shared_to: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """Average rigid transform (R, t) mapping the 'from' frame onto the 'to'
    frame, given 1+ shared anchor poses (center, rotation) expressed in both.
    Rotation averaged via quaternion mean, translation directly -- used by
    both run_windowed_reconstruction and run_windowed_reconstruction_full_pool
    to align one window's DA3 output onto the previous window's frame."""
    from scipy.spatial.transform import Rotation

    Rs, ts = [], []
    for (c_from, r_from), (c_to, r_to) in zip(shared_from, shared_to):
        R = r_to @ r_from.T
        Rs.append(R)
        ts.append(c_to - R @ c_from)
    quats = np.array([Rotation.from_matrix(R).as_quat() for R in Rs])
    quats *= np.sign(quats @ quats[0])[:, None]
    return Rotation.from_quat(quats.mean(axis=0)).as_matrix(), np.mean(ts, axis=0)


def save_da3_pointcloud(points: np.ndarray, colors: np.ndarray, path: str) -> str:
    """Thin public wrapper over components.Saver -- for callers outside this
    package (e.g. street_builder's windowed reconstruction, which transforms
    and merges several run_da3_pointcloud_with_poses results before saving
    one final output) that need to save a raw point cloud without reaching
    into this package's internal components.* modules directly."""
    Saver.save_point_cloud(points, path, colors=colors)
    return path


def _solo_score(da3: "DA3Model", path: str, tag: str, views_base: str, dist_thresh: float, angle_thresh: float, step_degrees: int = 20) -> int:
    """Extract one candidate's own ~18 DA3 view-slices and score it alone (no
    other pano in the batch) via the given already-loaded DA3Model, returning
    how many views survive the consensus filter. Shared by score_candidates
    and run_windowed_reconstruction, so both use the exact same scoring
    logic regardless of whether they load their own DA3Model for one call or
    reuse one across many calls in a single GPU session.

    step_degrees: matters more than it might look -- a candidate's keep-rate
    depends on its own per-pano consensus (median center, mean rotation
    across its own slices), and fewer slices makes that consensus noisier,
    not just less redundant. Scoring at a different step than the actual
    reconstruction call risks a mismatch (a candidate whose consensus is
    robust at 18 slices might not be at 8), so this should generally match
    whatever step_degrees the caller's reconstruction step uses.
    """
    d = os.path.join(views_base, tag)
    os.makedirs(d, exist_ok=True)
    views = extract_views_for_da3(path, d, prefix=f"{tag}_", pano_id=0, step_degrees=step_degrees)
    filtered_views, _ = da3.process_views(views, dist_thresh=dist_thresh, angle_thresh=angle_thresh)
    return len(filtered_views)


def _run_da3(
    target_depth_path: str,
    support_paths: list[str],
    cfg: "PipelineConfig",
    views_base: str,
    da3: "DA3Model | None" = None,
    dist_thresh: float = 0.2,
    angle_thresh: float = 1,
    step_degrees: int = 20,
):
    """Run the entire DA3 side of the pipeline: slice target + support panos
    into views, run DA3's joint multi-view pose+depth inference, and
    backproject to world-space points/colors. Shared by run() (depth/scale
    support for SHARP) and run_da3_pointcloud() (the entire output).

    da3: reuse an already-loaded DA3Model instead of loading (and deleting)
    a fresh one -- used by run_windowed_reconstruction, which makes several
    of these calls in one GPU session and would otherwise reload the model
    each time. Default None preserves the original behavior (load, use,
    delete) for every other caller.

    step_degrees: yaw spacing between slices (default 20 -- matches
    extract_views_for_da3's own default, i.e. 18 slices/pano at 90 HFOV,
    ~78% overlap between neighbors). Coarser values (e.g. 45 -> 8
    slices/pano) trade per-pano slice redundancy for a lower image count at
    the same viewpoint coverage -- exposed for experimenting with that
    tradeoff, not used by default anywhere.
    """
    all_views = []
    for i, path in enumerate([target_depth_path, *support_paths]):
        da3_dir = os.path.join(views_base, f"views_pano_{i}_da3")
        os.makedirs(da3_dir, exist_ok=True)
        all_views.extend(extract_views_for_da3(path, da3_dir, prefix=f"pano_{i}_", pano_id=i, step_degrees=step_degrees))

    owns_da3 = da3 is None
    if owns_da3:
        da3 = DA3Model(cfg.da3_model)
    filtered_views, da3_result = da3.process_views(all_views, dist_thresh=dist_thresh, angle_thresh=angle_thresh)
    merged_pts, merged_cols, per_pano_pts, per_pano_cols = backproject_views_to_pcd(
        filtered_views, da3_result
    )
    if owns_da3:
        del da3
        torch.cuda.empty_cache()
    return filtered_views, da3_result, merged_pts, merged_cols, per_pano_pts, per_pano_cols


def _run_da3_gs_pipeline(target_depth_path: str, output_dir: str, cfg: "PipelineConfig") -> None:
    """DA3 GS backend: skips SHARP entirely. Extracts perspective views from the
    target panorama and passes them directly to DA3 with infer_gs=True, which
    produces a unified 3DGS in one feed-forward pass."""
    from depth_anything_3.utils.gsply_helpers import save_gaussian_ply

    with tempfile.TemporaryDirectory() as views_dir:
        views = extract_views_for_da3(target_depth_path, views_dir, prefix="da3gs_", pano_id=0)
        print(f"  Extracted {len(views)} views for DA3 GS from {target_depth_path}")

        da3 = DA3Model(cfg.da3_model)
        prediction = da3.model.inference(
            [v.path for v in views],
            infer_gs=True,
            export_format="mini_npz",
        )

    final_path = os.path.join(output_dir, "final_output.ply")
    ctx_depth = torch.from_numpy(prediction.depth).unsqueeze(-1).to(prediction.gaussians.means)
    save_gaussian_ply(prediction.gaussians, final_path, ctx_depth=ctx_depth)
    print(f"DA3 GS pipeline complete: {final_path}")


def load_panorama_folder(folder_path: str) -> tuple[list[str], list[str | None], list[dict]]:
    """Load panoramas from a folder containing metadata.json and pano_{id}.jpg files."""
    with open(os.path.join(folder_path, "metadata.json")) as f:
        metadata = json.load(f)

    panorama_paths = []
    depth_paths = []
    for entry in metadata:
        pid = entry["id"]
        panorama_paths.append(os.path.join(folder_path, f"pano_{pid}.jpg"))
        depth_file = os.path.join(folder_path, f"pano_{pid}_depth.npy")
        depth_paths.append(depth_file if os.path.exists(depth_file) else None)

    return panorama_paths, depth_paths, metadata


class Pipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def run(
        self,
        target_appearance_path: str,
        output_dir: str,
        target_depth_path: str | None = None,
        support_paths: list[str] | None = None,
        save_da3_pointcloud: bool = False,
    ) -> Gaussians3D:
        """Run the full pipeline: align, process, and merge Gaussian splats for one target pano.

        Args:
            target_appearance_path: Image used to produce the 3DGS (SHARP input).
                            May be an edited/relit version of the panorama.
            output_dir: Directory to write outputs.
            target_depth_path: Image used for the target's DA3 depth/pose.
                            Defaults to target_appearance_path. Pass the original,
                            unedited panorama here when target_appearance_path has
                            been relit (e.g. day→night) — DA3 struggles to match
                            features across dark scenes.
            support_paths: Nearby panoramas used only as DA3 depth/pose context.
            save_da3_pointcloud: Also export the raw DA3 point cloud (target +
                            support panos merged) as da3_pointcloud.ply, alongside
                            the Gaussian splat. Independent of cfg.debug.

        Returns:
            Merged Gaussian splat, anchored so the target's capture point lands
            at (0, 0, 0) (also saved as final_output.ply).
        """
        cfg = self.config
        debug = cfg.debug
        target_depth_path = target_depth_path or target_appearance_path
        support_paths = support_paths or []
        print(
            f"Starting pipeline for target + {len(support_paths)} support panoramas "
            f"| Backend: {cfg.gs_backend} | Debug: {debug}"
        )

        os.makedirs(output_dir, exist_ok=True)

        if cfg.gs_backend == "da3":
            _run_da3_gs_pipeline(target_depth_path, output_dir, cfg)
            return None
        saver = Saver() if (debug or save_da3_pointcloud) else None

        with contextlib.ExitStack() as stack:
            # In debug mode, write view slices into output_dir so they persist.
            # Otherwise use a temp dir that is deleted automatically when the run finishes.
            if debug:
                views_base = output_dir
            else:
                views_base = stack.enter_context(tempfile.TemporaryDirectory())

            sharp_dir = os.path.join(views_base, "views_target_sharp")
            os.makedirs(sharp_dir, exist_ok=True)
            all_sharp_views = extract_views(
                target_appearance_path,
                sharp_dir,
                overlap_degrees=20,
                slice_count=cfg.slice_count,
                prefix="pano_0_",
                panorama_depth=None,
                pano_id=0,
                include_sky=cfg.include_sky,
            )

            print("--- Step: DA3 Global Pose Processing ---")
            (
                filtered_da3_views,
                da3_result,
                da3_pts,
                da3_cols,
                da3_pts_per_pano,
                _da3_cols_per_pano,
            ) = _run_da3(target_depth_path, support_paths, cfg, views_base)
            pano_poses = da3_result.pano_poses

            if da3_pts is not None:
                if debug:
                    print("--- Step: Saving DA3 Debug PCDs ---")
                    saver.save_point_cloud(
                        da3_pts,
                        os.path.join(output_dir, "da3_debug_consistency.ply"),
                        colors=da3_cols,
                    )
                    for pid, pts in da3_pts_per_pano.items():
                        saver.save_point_cloud(
                            pts, os.path.join(output_dir, f"da3_debug_pano_{pid}.ply")
                        )
                if save_da3_pointcloud:
                    saver.save_point_cloud(
                        da3_pts, os.path.join(output_dir, "da3_pointcloud.ply"), colors=da3_cols
                    )

            n_da3_clean = len(filtered_da3_views)
            del da3_result, filtered_da3_views, da3_cols, da3_pts
            torch.cuda.empty_cache()

            print(f"Generating splats for {len(all_sharp_views)} views of the target pano")

            print("--- Step: Splat Generation (SHARP) ---")
            gs_generator = SplatGenerator(cfg.sharp_model)
            splat_out_dir = os.path.join(output_dir, "gs") if debug else None
            gaussian_list = gs_generator.generate_from_views(all_sharp_views, output_dir=splat_out_dir)
            del gs_generator
            torch.cuda.empty_cache()

            # ExitStack closes here — temp dirs deleted after SHARP reads view slices
            # but before we write final PLYs (which go to output_dir, not views_base).

        print("--- Step: Splat Processing (Alignment/Merge) ---")
        # Flatten per-pano DA3 points into one global cloud (used by both alignment
        # paths and the floor view).
        all_da3_pts = (
            np.concatenate(
                [pts for pts in da3_pts_per_pano.values() if pts is not None], axis=0
            )
            if da3_pts_per_pano
            else None
        )

        processor = SplatProcessor(
            num_z_slabs=cfg.num_z_slabs,
            num_fov_slabs=cfg.num_fov_slabs,
            smooth_sigma_m=cfg.smooth_sigma_m,
            smooth_sigma_fov=cfg.smooth_sigma_fov,
            floor_keep_fraction=cfg.floor_keep_fraction,
            min_depth_coverage=cfg.min_depth_coverage,
            align_depth=cfg.align_depth,
            near_depth=cfg.near_depth,
            sky_depth=cfg.sky_depth,
        )
        merged_splat = processor.process(
            all_sharp_views,
            gaussian_list,
            pano_poses=pano_poses,
            all_da3_pts=all_da3_pts,
            scale_mode=cfg.scale_mode,
            n_da3_clean=n_da3_clean,
        )

        ref_view = all_sharp_views[0]
        final_path = os.path.join(output_dir, "final_output.ply")
        save_ply(
            merged_splat,
            f_px=ref_view.focal_px,
            image_shape=(ref_view.height, ref_view.width),
            path=final_path,
        )
        print(f"Pipeline complete: {final_path}")

        del gaussian_list, all_sharp_views, processor
        torch.cuda.empty_cache()
        return merged_splat

    def run_da3_pointcloud(
        self,
        target_depth_path: str,
        output_dir: str,
        support_paths: list[str] | None = None,
        step_degrees: int = 20,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run only the DA3 half of the pipeline: no SHARP, no Gaussians.

        target + support panos are fed to DA3 jointly for pose/depth
        estimation, and all of their points are backprojected and merged into
        the output — DA3's own multi-view consensus decides how well they
        line up, same as the pano_poses used to align SHARP's splats in .run().

        Useful as raw material for non-photoreal art (voxel grids, low-poly
        meshing, etc.) instead of a full Gaussian splat, where DA3's point
        cloud alone is easier to control than SHARP's splats.

        step_degrees: see _run_da3 -- default 20 (18 slices/pano) matches
        prior behavior; exposed here for experimenting with slice
        density/redundancy vs. image count.

        Returns:
            (points, colors): (N, 3) float32 world-space points and (N, 3)
            float colors in [0, 1], merged across target + support panos.
            Also saved as da3_pointcloud.ply in output_dir.
        """
        cfg = self.config
        support_paths = support_paths or []
        print(f"Starting DA3-only pipeline for target + {len(support_paths)} support panoramas")

        os.makedirs(output_dir, exist_ok=True)

        with tempfile.TemporaryDirectory() as views_base:
            _, _, pts, cols, _, _ = _run_da3(target_depth_path, support_paths, cfg, views_base, step_degrees=step_degrees)

        if pts is None:
            raise RuntimeError(
                "No usable views survived DA3 filtering "
                "(check the 'Filtering view ...' logs above for dist/angle deviation)."
            )

        final_path = os.path.join(output_dir, "da3_pointcloud.ply")
        Saver.save_point_cloud(pts, final_path, colors=cols)
        print(f"DA3 point cloud pipeline complete: {final_path}")
        return pts, cols

    def run_da3_pointcloud_sweep(
        self,
        target_depth_path: str,
        output_dir: str,
        threshold_levels: list[tuple[float, float]],
        support_paths: list[str] | None = None,
    ) -> list[str | None]:
        """Debug/diagnostic variant of run_da3_pointcloud: runs DA3 inference
        once and saves one point cloud per (dist_thresh, angle_thresh) pair in
        threshold_levels, to compare how much the consensus filter actually
        matters without paying for a repeated GPU forward pass per level.

        Returns one path per threshold level (None for a level where no views
        survived), same order as threshold_levels.
        """
        cfg = self.config
        support_paths = support_paths or []
        print(
            f"Starting DA3 filter sweep ({len(threshold_levels)} levels) for "
            f"target + {len(support_paths)} support panoramas"
        )

        os.makedirs(output_dir, exist_ok=True)

        with tempfile.TemporaryDirectory() as views_base:
            all_views = []
            for i, path in enumerate([target_depth_path, *support_paths]):
                da3_dir = os.path.join(views_base, f"views_pano_{i}_da3")
                os.makedirs(da3_dir, exist_ok=True)
                all_views.extend(extract_views_for_da3(path, da3_dir, prefix=f"pano_{i}_", pano_id=i))

            da3 = DA3Model(cfg.da3_model)
            results = da3.process_views_sweep(all_views, threshold_levels)
            del da3
            torch.cuda.empty_cache()

        out_paths = []
        for (dist_thresh, angle_thresh), (filtered_views, da3_result) in zip(threshold_levels, results):
            if not filtered_views:
                print(f"Sweep level dist={dist_thresh} angle={angle_thresh}: no views survived, skipping.")
                out_paths.append(None)
                continue
            pts, cols, _, _ = backproject_views_to_pcd(filtered_views, da3_result)
            name = f"da3_pointcloud_d{dist_thresh}_a{angle_thresh}.ply"
            path = os.path.join(output_dir, name)
            Saver.save_point_cloud(pts, path, colors=cols)
            out_paths.append(path)

        print(f"DA3 filter sweep complete: {sum(p is not None for p in out_paths)}/{len(threshold_levels)} levels produced output")
        return out_paths

    def score_candidates(
        self,
        candidate_paths: list[str],
        dist_thresh: float = 0.2,
        angle_thresh: float = 1,
        step_degrees: int = 20,
    ) -> list[int]:
        """Solo self-consistency score for each candidate pano: extract just
        that pano's own ~18 DA3 view-slices and run them through DA3 alone (no
        other pano in the batch), then count how many survive the consensus
        filter. One DA3Model load shared across all candidates -- each
        candidate only costs one small forward pass, not a full model reload.

        Used to rank support-pano candidates by how internally coherent DA3
        finds them, before picking which ones to actually reconstruct with,
        instead of picking by raw distance alone.

        step_degrees: see _solo_score -- should generally match whatever
        step_degrees the caller's later reconstruction call uses, since a
        candidate's consensus robustness (and therefore its score) shifts
        with slice count.

        Returns one keep-count per candidate, same order as candidate_paths.
        """
        cfg = self.config
        da3 = DA3Model(cfg.da3_model)
        with tempfile.TemporaryDirectory() as views_base:
            scores = [
                _solo_score(da3, path, f"score_{i}", views_base, dist_thresh, angle_thresh, step_degrees=step_degrees)
                for i, path in enumerate(candidate_paths)
            ]
        del da3
        torch.cuda.empty_cache()
        return scores

    def run_windowed_reconstruction(
        self,
        windows: list[list[tuple[str, str, float, float]]],
        boundary_coords: list[tuple[float, float]],
        final_count: int = 4,
        forced_overlap: int = 2,
        dist_thresh: float = 0.2,
        angle_thresh: float = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Chunk+connect multi-window DA3 reconstruction, entirely within ONE
        DA3Model load reused across every window's scoring AND reconstruction.

        This consolidation matters specifically on ZeroGPU: each separate
        @spaces.GPU call is its own fresh GPU acquisition (and, since this
        package's Pipeline doesn't cache a DA3Model across calls, a fresh
        ~35GB model reload). Calling per-window helpers (what used to be
        score_candidates + run_da3_pointcloud_with_poses) separately, once per
        window, from the caller side hit ZeroGPU's proxy-token lifetime after
        only 2 windows (4 sequential GPU calls) in practice -- hence this
        single entry point doing the whole multi-window job in one GPU call.

        windows: one list per window, each entry (label, path, lat, lon) --
        candidates already downloaded, from whatever domain-specific source
        (street_builder's Google/Apple pool gathering, in the current caller;
        this package has no dependency on that -- it just needs path+coords).

        boundary_coords: len(windows)-1; boundary_coords[i] = the (lat, lon)
        of the node shared between window i and window i+1, used to decide
        which of window i's winners get carried forward into window i+1 as
        forced overlap.

        Per window: solo-scores every candidate (self-consistency keep-rate,
        see DA3Model.process_views) and picks the top final_count -- except
        forced_overlap of them (for every window after the first), which are
        forced to be the previous window's own winners closest to that
        window's boundary node. That forcing does two things at once:
        guarantees those slots are already-vetted good nodes, and guarantees
        the two windows share a literal identical image, which is what makes
        rigid alignment between them valid at all. Runs DA3 on each window's
        winners, rigid-aligns (rotation+translation only -- DA3's metric scale
        is trusted consistent within this one session, so no scale term is
        solved for) onto a running global frame anchored on the first window,
        via the shared images' poses in both windows, and merges.

        Returns (points, colors) merged across all windows in that global
        frame.
        """
        cfg = self.config
        da3 = DA3Model(cfg.da3_model)

        global_pts = global_cols = None
        global_R, global_t = np.eye(3), np.zeros(3)
        prev_winners = prev_poses = None

        try:
            with tempfile.TemporaryDirectory() as views_base:
                for window_idx, pool in enumerate(windows):
                    scores = [
                        _solo_score(da3, path, f"w{window_idx}_score{i}", views_base, dist_thresh, angle_thresh)
                        for i, (label, path, lat, lon) in enumerate(pool)
                    ]
                    ranked = [c for c, _ in sorted(zip(pool, scores), key=lambda x: x[1], reverse=True)]
                    print(f"Window {window_idx} candidate scores: {list(zip((c[0] for c in pool), scores))}")

                    if prev_winners is None:
                        winners = ranked[:final_count]
                        forced_paths = []
                    else:
                        b_lat, b_lon = boundary_coords[window_idx - 1]
                        forced = sorted(prev_winners, key=lambda c: _haversine_m(b_lat, b_lon, c[2], c[3]))[:forced_overlap]
                        forced_paths = [c[1] for c in forced]
                        new_picks = [c for c in ranked if c[1] not in forced_paths][:final_count - len(forced)]
                        winners = forced + new_picks

                    if len(winners) < 2:
                        raise RuntimeError(f"Window {window_idx} has too few usable candidates for multi-view reconstruction.")
                    print(f"Window {window_idx} reconstructing with: {[c[0] for c in winners]}")

                    window_views_dir = os.path.join(views_base, f"w{window_idx}_recon")
                    os.makedirs(window_views_dir, exist_ok=True)
                    filtered_views, da3_result, pts, cols, _, _ = _run_da3(
                        winners[0][1],
                        [c[1] for c in winners[1:]],
                        cfg,
                        window_views_dir,
                        da3=da3,
                        dist_thresh=dist_thresh,
                        angle_thresh=angle_thresh,
                    )
                    if not filtered_views:
                        raise RuntimeError(f"Window {window_idx}: no views survived DA3 filtering.")
                    pano_poses = da3_result.pano_poses

                    if window_idx == 0:
                        global_pts, global_cols = pts, cols
                    else:
                        this_idx = {c[1]: i for i, c in enumerate(winners)}
                        prev_idx = {c[1]: i for i, c in enumerate(prev_winners)}
                        try:
                            shared_from = [(pano_poses[this_idx[p]]["center"], pano_poses[this_idx[p]]["rotation"]) for p in forced_paths]
                            shared_to = [(prev_poses[prev_idx[p]]["center"], prev_poses[prev_idx[p]]["rotation"]) for p in forced_paths]
                        except KeyError:
                            raise RuntimeError(
                                f"Window {window_idx}: a forced overlap anchor's views were entirely "
                                "filtered out by DA3's consensus check in one of the two windows, so "
                                "there's no pose to align on."
                            )
                        local_R, local_t = _rigid_align(shared_from, shared_to)
                        global_R, global_t = global_R @ local_R, global_R @ local_t + global_t
                        global_pts = np.concatenate([global_pts, pts @ global_R.T + global_t], axis=0)
                        global_cols = np.concatenate([global_cols, cols], axis=0)

                    prev_winners, prev_poses = winners, pano_poses
        finally:
            del da3
            torch.cuda.empty_cache()

        return global_pts, global_cols

    def run_windowed_reconstruction_full_pool(
        self,
        windows: list[list[tuple[str, str, float, float]]],
        dist_thresh: float = 0.2,
        angle_thresh: float = 1,
        step_degrees: int = 20,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Chunk+connect variant of run_windowed_reconstruction that skips
        solo-scoring and down-selection entirely: every window's FULL
        candidate list goes straight into that window's DA3 call. Tests
        whether picking only the top-scoring candidates (what
        run_windowed_reconstruction does) is actually necessary once
        windowing has already bounded each window to a tight geographic
        range, or whether that range alone is enough.

        windows: one list per window, each entry (label, path, lat, lon) --
        same shape as run_windowed_reconstruction, but every entry here goes
        into the reconstruction, none are scored or dropped.

        Alignment between window i and i+1 doesn't need an explicit forced-
        carryover step like run_windowed_reconstruction's: since adjacent
        windows are constructed to share one raw chain node (see the
        caller), and both windows' full candidate lists independently
        include that node's own image and its nearest-K neighbors, those
        candidates already appear in both windows' lists by construction --
        matched here by path. Alignment uses whichever of those naturally-
        shared candidates (by path) have valid poses in both windows' DA3
        results; as few as 1 is enough for a full rigid solve, so this only
        fails if literally none of them survive DA3's filter in both.

        Returns (points, colors) merged across all windows in one global
        frame.
        """
        cfg = self.config
        da3 = DA3Model(cfg.da3_model)

        global_pts = global_cols = None
        global_R, global_t = np.eye(3), np.zeros(3)
        prev_pool = prev_poses = None

        try:
            with tempfile.TemporaryDirectory() as views_base:
                for window_idx, pool in enumerate(windows):
                    print(f"Window {window_idx} reconstructing with full pool ({len(pool)}): {[c[0] for c in pool]}")

                    window_views_dir = os.path.join(views_base, f"w{window_idx}_recon")
                    os.makedirs(window_views_dir, exist_ok=True)
                    filtered_views, da3_result, pts, cols, _, _ = _run_da3(
                        pool[0][1],
                        [c[1] for c in pool[1:]],
                        cfg,
                        window_views_dir,
                        da3=da3,
                        dist_thresh=dist_thresh,
                        angle_thresh=angle_thresh,
                        step_degrees=step_degrees,
                    )
                    if not filtered_views:
                        raise RuntimeError(f"Window {window_idx}: no views survived DA3 filtering.")
                    pano_poses = da3_result.pano_poses

                    if window_idx == 0:
                        global_pts, global_cols = pts, cols
                    else:
                        this_idx = {c[1]: i for i, c in enumerate(pool)}
                        prev_idx = {c[1]: i for i, c in enumerate(prev_pool)}
                        shared_from, shared_to = [], []
                        for path in this_idx:
                            if path not in prev_idx:
                                continue
                            ti, pi = this_idx[path], prev_idx[path]
                            if ti in pano_poses and pi in prev_poses:
                                shared_from.append((pano_poses[ti]["center"], pano_poses[ti]["rotation"]))
                                shared_to.append((prev_poses[pi]["center"], prev_poses[pi]["rotation"]))
                        if not shared_from:
                            raise RuntimeError(
                                f"Window {window_idx}: none of the candidates shared with the previous "
                                "window survived DA3's consensus check in both, so there's no pose to align on."
                            )
                        local_R, local_t = _rigid_align(shared_from, shared_to)
                        global_R, global_t = global_R @ local_R, global_R @ local_t + global_t
                        global_pts = np.concatenate([global_pts, pts @ global_R.T + global_t], axis=0)
                        global_cols = np.concatenate([global_cols, cols], axis=0)

                    prev_pool, prev_poses = pool, pano_poses
        finally:
            del da3
            torch.cuda.empty_cache()

        return global_pts, global_cols
