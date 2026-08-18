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
    (forced-overlap selection by distance to a window's boundary node)."""
    from math import atan2, cos, radians, sin, sqrt

    earth_radius_m = 6371000.0
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * earth_radius_m * atan2(sqrt(a), sqrt(1 - a))


def _rigid_align(shared_from: list[tuple[np.ndarray, np.ndarray]], shared_to: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """Average rigid transform (R, t) mapping the 'from' frame onto the 'to'
    frame, given 1+ shared anchor poses (center, rotation) expressed in both.
    Rotation averaged via quaternion mean, translation directly -- used by
    run_windowed_reconstruction and run_greedy_pass_reconstruction to align
    one window's DA3 output onto the previous window's frame.

    r_from/r_to are world-to-pano rotations for the SAME physical anchor,
    expressed in each call's own arbitrary world frame (v_pano = r @ v_world).
    For a direction to agree either way it's expressed: r_from @ v_from ==
    r_to @ v_to, and v_to = R @ v_from, so r_from = r_to @ R, i.e.
    R = r_to^-1 @ r_from = r_to.T @ r_from (rotations are orthogonal).
    Verified against synthetic ground truth (random Q, reconstructed to
    <0.001 deg) -- an earlier r_to @ r_from.T was off by ~90 deg on real
    data, which is what caused pathfind reconstructions to visibly fold
    into a '*' shape instead of following the real street."""
    from scipy.spatial.transform import Rotation

    Rs, ts = [], []
    for (c_from, r_from), (c_to, r_to) in zip(shared_from, shared_to):
        R = r_to.T @ r_from
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
    views = extract_views_for_da3(path, d, prefix=f"{tag}_", pano_id=os.path.basename(path), step_degrees=step_degrees)
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
        all_views.extend(extract_views_for_da3(path, da3_dir, prefix=f"pano_{i}_", pano_id=os.path.basename(path), step_degrees=step_degrees))

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

    def run_greedy_pass_reconstruction(
        self,
        node_candidates: list[list[tuple[str, str, str, float, float]]],
        try_order: list[list[str]],
        keep_rate_threshold: float = 0.5,
        max_attempts_per_position: int = 3,
        dist_thresh: float = 0.2,
        angle_thresh: float = 1,
        step_degrees: int = 20,
    ) -> list[tuple[np.ndarray, np.ndarray, tuple[int, int], str]]:
        """Greedy same-date sliding-window reconstruction, entirely within
        ONE DA3Model load (same ZeroGPU-consolidation reason as
        run_windowed_reconstruction).

        Unlike run_windowed_reconstruction (which always uses a solo
        self-consistency score to pick candidates, then hopes they correlate
        once combined), every window here is graded by the actual thing that
        matters: a real 2-candidate DA3 call, checking how many of each
        side's own view-slices survive the consensus filter. This session
        found solo score doesn't reliably predict that outcome in either
        direction, so there's no scoring step to skip here -- the pairwise
        call IS the grading.

        node_candidates: one list per street node (in order), each entry
        (date, label, path, lat, lon) -- e.g. ("2024-07", "apple:123",
        "/path.jpg", 1.23, 103.4). `date` identifies which capture pass a
        candidate belongs to (Apple and Google both use the same date-string
        format, see the caller); candidates from different nodes with the
        same date are treated as the same pass/visit, regardless of source
        -- an Apple candidate and a Google candidate can share a "pass" if
        they land on the same date. That's untested as a general rule, which
        is exactly why every window is still graded by a real pairwise DA3
        call, never trusted from the date match alone.

        try_order: one list per node, of dates in the order to attempt them
        when starting a *new* segment at that node -- precomputed by the
        caller from cheap metadata (coverage ranking), not derived here.
        This method never re-ranks; it only walks forward using whatever
        order it's given, capped at max_attempts_per_position attempts per
        position.

        Walk: at each position i, build a 2-node window from node i and
        node i+1 using the same date on both sides. If a segment is already
        in progress, its active date is tried first (continuity is
        preferred over switching to a "better" pass, since switching can't
        be rigid-aligned -- there's no shared image between two different
        passes). Otherwise (or if the active date fails or isn't available
        at this position), try try_order[i]'s dates in order, skipping ones
        missing at either node, capped at max_attempts_per_position. A
        window is "healthy" if both sides kept at least keep_rate_threshold
        of their own view-slices (DA3Result.pano_keep_counts). On success:
        advance, and if continuing the active date, rigid-align this
        window's node-i pose onto where node i was placed in the previous
        window (the two windows share that exact image). On failure at
        every attempted date: close out the current segment (if any) and
        start a fresh search at i+1.

        Returns a list of segments, each (points, colors, (start_node_idx,
        end_node_idx), date) -- deliberately NOT one merged cloud, since a
        street with no single pass covering it end-to-end is expected to
        break into disconnected segments, not something to force into one
        result.
        """
        cfg = self.config
        da3 = DA3Model(cfg.da3_model)

        node_dicts = [
            {date: (label, path, lat, lon) for date, label, path, lat, lon in candidates}
            for candidates in node_candidates
        ]

        segments = []
        seg_pts = seg_cols = None
        seg_R, seg_t = np.eye(3), np.zeros(3)
        seg_start = None
        active_key = None
        prev_pair_poses = None  # [(center, rotation) for node i, node i+1] of the last committed window

        def healthy(da3_result, id_a, id_b):
            # DA3Model's pano_id (both DA3Result.pano_keep_counts and
            # .pano_poses) is os.path.basename(path), not a positional
            # index -- see _run_da3's extract_views_for_da3 calls.
            kept_a, total_a = da3_result.pano_keep_counts.get(id_a, (0, 1))
            kept_b, total_b = da3_result.pano_keep_counts.get(id_b, (0, 1))
            return (kept_a / total_a) >= keep_rate_threshold and (kept_b / total_b) >= keep_rate_threshold

        try:
            with tempfile.TemporaryDirectory() as views_base:
                i = 0
                while i < len(node_candidates) - 1:
                    attempts = []
                    if active_key is not None and active_key in node_dicts[i] and active_key in node_dicts[i + 1]:
                        attempts.append(active_key)
                    for key in try_order[i]:
                        if len(attempts) >= max_attempts_per_position:
                            break
                        if key == active_key or key in attempts:
                            continue
                        if key not in node_dicts[i] or key not in node_dicts[i + 1]:
                            continue
                        attempts.append(key)

                    committed_key = None
                    committed_id_a = committed_id_b = None
                    for key in attempts:
                        _, path_a, _, _ = node_dicts[i][key]
                        _, path_b, _, _ = node_dicts[i + 1][key]
                        id_a, id_b = os.path.basename(path_a), os.path.basename(path_b)
                        window_dir = os.path.join(views_base, f"pos{i}_{key}".replace("/", "_"))
                        os.makedirs(window_dir, exist_ok=True)
                        filtered_views, da3_result, pts, cols, _, _ = _run_da3(
                            path_a, [path_b], cfg, window_dir,
                            da3=da3, dist_thresh=dist_thresh, angle_thresh=angle_thresh, step_degrees=step_degrees,
                        )
                        if healthy(da3_result, id_a, id_b):
                            committed_key = key
                            committed_id_a, committed_id_b = id_a, id_b
                            break

                    if committed_key is None:
                        if seg_pts is not None:
                            segments.append((seg_pts, seg_cols, (seg_start, i), active_key))
                        seg_pts = seg_cols = None
                        active_key = None
                        prev_pair_poses = None
                        i += 1
                        continue

                    pano_poses = da3_result.pano_poses
                    node_poses = [
                        (pano_poses[committed_id_a]["center"], pano_poses[committed_id_a]["rotation"]),
                        (pano_poses[committed_id_b]["center"], pano_poses[committed_id_b]["rotation"]),
                    ]

                    if committed_key == active_key and prev_pair_poses is not None:
                        local_R, local_t = _rigid_align([node_poses[0]], [prev_pair_poses[1]])
                        seg_R, seg_t = seg_R @ local_R, seg_R @ local_t + seg_t
                        seg_pts = np.concatenate([seg_pts, pts @ seg_R.T + seg_t], axis=0)
                        seg_cols = np.concatenate([seg_cols, cols], axis=0)
                    else:
                        if seg_pts is not None:
                            segments.append((seg_pts, seg_cols, (seg_start, i), active_key))
                        seg_pts, seg_cols = pts, cols
                        seg_R, seg_t = np.eye(3), np.zeros(3)
                        seg_start = i

                    active_key = committed_key
                    prev_pair_poses = node_poses
                    i += 1

                if seg_pts is not None:
                    segments.append((seg_pts, seg_cols, (seg_start, len(node_candidates) - 1), active_key))
        finally:
            del da3
            torch.cuda.empty_cache()

        return segments

    def run_pathfind_reconstruction(
        self,
        nodes: list[tuple[str, str, float, float, str]],
        edges: dict,
        start_lat: float,
        start_lon: float,
        goals: list[tuple[float, float]],
        target_hop_m: float = 10.0,
        hop_weight: float = 1.0,
        start_zone_m: float = 5.0,
        goal_tolerance_m: float = 15.0,
        max_tests_per_date: int = 50,
        keep_rate_threshold: float = 0.25,
        dist_thresh: float = 0.2,
        angle_thresh: float = 1,
        step_degrees: int = 20,
        date_order: list[str] | None = None,
        max_segments: int = 5,
    ) -> list[tuple[np.ndarray, np.ndarray, list, str, bool]]:
        """Best-first search over a candidate graph, start -> every goal in
        `goals`, entirely within ONE DA3Model load (same ZeroGPU reason as
        the methods above).

        Inputs (all pre-downloaded by the caller -- no network here):
        - nodes: (key, path, lat, lon, date) per candidate pano.
        - edges: {key: [(other_key, dist_m), ...]} -- untested same-date hops
          from build_graph; this method runs the real DA3 test on each.
        - goals: real (lat, lon) points to reach -- more than one supports a
          branching selection (e.g. a junction with multiple arms): the
          search doesn't stop at the first one reached, it keeps growing the
          same frontier toward whatever's still outstanding, since a
          confirmed junction node already pushes ALL of its same-date
          neighbors onto the frontier -- covering every branch is what the
          existing best-first mechanics already do, this just stops it from
          quitting early.

        Search: one date at a time, in date_order if given (falls back to
        whatever order roots_by_date built in for any date not listed --
        not a stable order, see the code comment where it's used). Each
        date is its own independent best-first search (own frontier, own
        confirmed tree -- a single reconstruction must be one date, no
        shared image to align across dates), each starting from the FULL
        outstanding goal set for this segment -- one date reaching a goal
        doesn't mean a different date's own (separate) reconstruction also
        covers it. Frontier edges scored by
        dist(child, nearest outstanding goal) + hop_weight * |hop -
        target_hop_m| (lower first). Seed: every node within start_zone_m
        of start for that date. Stitch each passing edge onto the tree via
        _rigid_align on the shared (already-confirmed) parent pano, same
        math as greedy's walk.

        Per segment, the date that covers the most goals (ties broken by
        closest remaining distance) wins; its actually-covered goals are
        removed from the overall outstanding set.

        Multi-segment: if goals remain outstanding after the best date's
        search, the confirmed node (from that winning date) closest to the
        nearest remaining goal becomes the start point for a new segment --
        same per-date search, any date, no date assumed to continue where
        the last one left off (that's exactly the case DA3 can't bridge on
        its own). Repeats up to max_segments times, or stops early if a new
        segment covers no further goals and makes no real progress toward
        the nearest remaining one (a genuine dead end, not just "this date
        ran out"). Segments are NOT stitched together here -- each is
        independently placed by DA3 in its own arbitrary frame; joining
        them (e.g. via real GPS + ICP) is a separate step done by the
        caller.

        Returns a list of (pts, cols, path_edges, date, reached_all) tuples,
        one per segment, in the order they were produced. reached_all is
        True only for the segment (if any) after which every goal was
        covered. Empty if the very first segment couldn't connect anything.
        """
        import heapq

        cfg = self.config
        node_by_key = {key: (path, lat, lon, date) for key, path, lat, lon, date in nodes}
        if not node_by_key or not goals:
            return []

        def gdist_point(lat, lon, goal):
            return _haversine_m(lat, lon, goal[0], goal[1])

        def nearest_goal_dist(key, remaining_idx):
            _, lat, lon, _ = node_by_key[key]
            return min(gdist_point(lat, lon, goals[gi]) for gi in remaining_idx)

        def score(child_key, hop, remaining_idx):
            return nearest_goal_dist(child_key, remaining_idx) + hop_weight * abs(hop - target_hop_m)

        def healthy(res, id_a, id_b):
            ka, ta = res.pano_keep_counts.get(id_a, (0, 1))
            kb, tb = res.pano_keep_counts.get(id_b, (0, 1))
            return (ka / ta) >= keep_rate_threshold and (kb / tb) >= keep_rate_threshold

        def roots_for(lat0, lon0):
            """Seed roots per date: every node within start_zone_m of
            (lat0, lon0), dates tried in the caller's ranked order (e.g.
            coverage-span rank) where given -- roots_by_date's own
            iteration order isn't guaranteed stable across runs otherwise
            (nodes arrives via a caller-side dict/set, whose order can
            depend on Python's per-process string hash seed)."""
            roots_by_date = {}
            for key, (_, lat, lon, date) in node_by_key.items():
                if _haversine_m(lat, lon, lat0, lon0) <= start_zone_m:
                    roots_by_date.setdefault(date, []).append(key)
            if date_order:
                ordered_dates = [d for d in date_order if d in roots_by_date]
                ordered_dates += [d for d in roots_by_date if d not in ordered_dates]
                roots_by_date = {d: roots_by_date[d] for d in ordered_dates}
            return roots_by_date

        def search_date(date, root_keys, da3, views_base, test_offset, remaining_idx):
            """Best-first search restricted to one date, against its own
            copy of remaining_idx (mutated in place -- caller reads it back
            afterward to see which goals THIS date's search covered).
            Returns (pts, cols, path_edges, confirmed, tests_done)."""
            frontier = []  # (score, seq, from_key, to_key, hop)
            seq = 0
            for root_key in root_keys:
                for other_key, hop in edges.get(root_key, []):
                    if node_by_key[other_key][3] != date:
                        continue
                    heapq.heappush(frontier, (score(other_key, hop, remaining_idx), seq, root_key, other_key, hop))
                    seq += 1

            confirmed = {}
            global_pts = global_cols = None
            path_edges = []
            tests = 0

            while frontier and tests < max_tests_per_date and remaining_idx:
                _, _, from_key, to_key, hop = heapq.heappop(frontier)
                if to_key in confirmed:
                    continue
                if path_edges and from_key not in confirmed:
                    continue

                path_a, path_b = node_by_key[from_key][0], node_by_key[to_key][0]
                id_a, id_b = os.path.basename(path_a), os.path.basename(path_b)
                test_dir = os.path.join(views_base, f"t{test_offset + tests}")
                os.makedirs(test_dir, exist_ok=True)
                _, res, pts, cols, _, _ = _run_da3(
                    path_a, [path_b], cfg, test_dir,
                    da3=da3, dist_thresh=dist_thresh, angle_thresh=angle_thresh, step_degrees=step_degrees,
                )
                tests += 1
                ok = healthy(res, id_a, id_b)
                hop_num = len(path_edges) + 1
                status = "OK" if ok else "FAIL, trying next candidate"
                print(f"[{date} hop {hop_num}] {from_key} -> {to_key} (hop {hop:.1f}m, {nearest_goal_dist(to_key, remaining_idx):.1f}m to nearest goal): {status}")
                if not ok:
                    continue

                pose_a = (res.pano_poses[id_a]["center"], res.pano_poses[id_a]["rotation"])
                pose_b = (res.pano_poses[id_b]["center"], res.pano_poses[id_b]["rotation"])

                def _log_pose(label, key, cam_center, cam_rot, seg_R, seg_t):
                    """cam_center/cam_rot: this pano's own pose as DA3 reported it
                    in the current test's local frame. seg_R/seg_t: that test
                    frame's transform into the global (tree-base) frame."""
                    from scipy.spatial.transform import Rotation
                    global_center = seg_R @ cam_center + seg_t
                    yaw = Rotation.from_matrix(seg_R @ cam_rot).as_euler('yxz', degrees=True)[0]
                    lc = np.array2string(cam_center, precision=3, suppress_small=True)
                    gc = np.array2string(global_center, precision=3, suppress_small=True)
                    _, real_lat, real_lon, _ = node_by_key[key]
                    print(f"  [{date}] pose {label} {key}: real_latlon=({real_lat:.7f}, {real_lon:.7f}), local_center={lc}, global_center={gc}, global_yaw={yaw:.1f}deg")

                def _log_raw_rotation(label, key, cam_center, cam_rot):
                    """The pano's consensus (center, rotation) exactly as DA3
                    reported it in THIS call's own local frame -- no seg_R/seg_t
                    applied, no accumulation. Full 3x3 rotation matrix, not just
                    a derived yaw, so it can be used directly (e.g. to render
                    which raw-panorama crop it implies) outside this run."""
                    lc = np.array2string(cam_center, precision=4, suppress_small=True)
                    rot = np.array2string(cam_rot, precision=4, suppress_small=True)
                    print(f"  [{date}] RAW pose {label} {key} (this call's own frame): center={lc}, rotation=\n{rot}")

                _log_raw_rotation("A", from_key, pose_a[0], pose_a[1])
                _log_raw_rotation("B", to_key, pose_b[0], pose_b[1])

                if not path_edges:
                    # First success for this date: this edge's frame is the tree's base.
                    confirmed[from_key] = {"seg_R": np.eye(3), "seg_t": np.zeros(3), "pose": pose_a}
                    confirmed[to_key] = {"seg_R": np.eye(3), "seg_t": np.zeros(3), "pose": pose_b}
                    global_pts, global_cols = pts, cols
                    _log_pose("A", from_key, pose_a[0], pose_a[1], np.eye(3), np.zeros(3))
                    _log_pose("B", to_key, pose_b[0], pose_b[1], np.eye(3), np.zeros(3))
                else:
                    # Align this edge's frame onto the confirmed parent, then to the tree base.
                    pf = confirmed[from_key]
                    local_R, local_t = _rigid_align([pose_a], [pf["pose"]])
                    seg_R = pf["seg_R"] @ local_R
                    seg_t = pf["seg_R"] @ local_t + pf["seg_t"]
                    confirmed[to_key] = {"seg_R": seg_R, "seg_t": seg_t, "pose": pose_b}
                    global_pts = np.concatenate([global_pts, pts @ seg_R.T + seg_t], axis=0)
                    global_cols = np.concatenate([global_cols, cols], axis=0)
                    _log_pose("B", to_key, pose_b[0], pose_b[1], seg_R, seg_t)
                path_edges.append((from_key, to_key))

                _, to_lat, to_lon, _ = node_by_key[to_key]
                newly_reached = [gi for gi in remaining_idx if gdist_point(to_lat, to_lon, goals[gi]) <= goal_tolerance_m]
                for gi in newly_reached:
                    remaining_idx.discard(gi)
                    print(f"  [{date}] reached goal {gi} {goals[gi]} ({len(remaining_idx)} still outstanding)")
                trail = " -> ".join([path_edges[0][0]] + [e[1] for e in path_edges])
                print(f"  [{date}] path so far: {trail}")

                if not remaining_idx:
                    break
                # Score against the now-possibly-smaller remaining_idx -- entries
                # already queued from before a goal was reached keep their older
                # (slightly stale) score, which only affects exploration order,
                # not correctness.
                for other_key, next_hop in edges.get(to_key, []):
                    if other_key in confirmed or node_by_key[other_key][3] != date:
                        continue
                    heapq.heappush(frontier, (score(other_key, next_hop, remaining_idx), seq, to_key, other_key, next_hop))
                    seq += 1

            return global_pts, global_cols, path_edges, confirmed, tests

        def confirmed_nearest(confirmed, remaining_idx):
            """(key, dist) of whichever confirmed node sits closest to any
            outstanding goal -- used both to rank which date's partial
            result is "best" and to pick the next segment's start point."""
            best_key, best_dist = None, float("inf")
            for key in confirmed:
                d = nearest_goal_dist(key, remaining_idx)
                if d < best_dist:
                    best_key, best_dist = key, d
            return best_key, best_dist

        da3 = DA3Model(cfg.da3_model)
        segments = []
        total_tests = 0
        overall_remaining = set(range(len(goals)))
        cur_lat, cur_lon = start_lat, start_lon
        best_progress = min(gdist_point(cur_lat, cur_lon, g) for g in goals)

        try:
            with tempfile.TemporaryDirectory() as views_base:
                for seg_i in range(max_segments):
                    roots_by_date = roots_for(cur_lat, cur_lon)
                    print(f"pathfind segment {seg_i + 1}: {sum(len(v) for v in roots_by_date.values())} roots across {len(roots_by_date)} dates from ({cur_lat:.6f}, {cur_lon:.6f}), {len(overall_remaining)} goal(s) outstanding")

                    best = None  # (pts, cols, path_edges, date, confirmed, remaining_after)
                    for date, root_keys in roots_by_date.items():
                        print(f"pathfind: trying date {date} ({len(root_keys)} roots)")
                        remaining_idx = set(overall_remaining)
                        pts, cols, path_edges, confirmed, tests = search_date(date, root_keys, da3, views_base, total_tests, remaining_idx)
                        total_tests += tests
                        if pts is None:
                            print(f"  [{date}] no hop succeeded")
                            continue
                        if not remaining_idx:
                            best = (pts, cols, path_edges, date, confirmed, remaining_idx)
                            break
                        _, best_dist_here = confirmed_nearest(confirmed, remaining_idx)
                        if best is None:
                            best = (pts, cols, path_edges, date, confirmed, remaining_idx)
                        else:
                            _, cur_best_dist = confirmed_nearest(best[4], best[5])
                            covers_more = len(remaining_idx) < len(best[5])
                            if covers_more or (len(remaining_idx) == len(best[5]) and best_dist_here < cur_best_dist):
                                best = (pts, cols, path_edges, date, confirmed, remaining_idx)

                    if best is None:
                        print(f"pathfind segment {seg_i + 1}: no date connected anything from this start -- stopping")
                        break

                    pts, cols, path_edges, date, confirmed, remaining_after = best
                    reached_all = not remaining_after
                    segments.append((pts, cols, path_edges, date, reached_all))
                    covered = overall_remaining - remaining_after
                    overall_remaining = remaining_after
                    print(f"pathfind segment {seg_i + 1}: {len(path_edges)} hops, date={date}, covered {len(covered)} goal(s), {len(overall_remaining)} still outstanding")

                    if not overall_remaining:
                        break
                    next_key, next_dist = confirmed_nearest(confirmed, overall_remaining)
                    if next_key is None or next_dist >= best_progress - 1.0:
                        print(f"pathfind: segment {seg_i + 1} made no real progress toward remaining goals -- stopping")
                        break
                    best_progress = next_dist
                    _, cur_lat, cur_lon, _ = node_by_key[next_key]
        finally:
            del da3
            torch.cuda.empty_cache()

        print(f"pathfind: {total_tests} attempts total, {len(segments)} segment(s), {len(overall_remaining)} goal(s) never reached")
        return segments
