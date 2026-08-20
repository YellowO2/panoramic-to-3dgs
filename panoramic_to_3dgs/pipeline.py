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


def test_edge_da3(
    path_a: str,
    path_b: str,
    cfg: "PipelineConfig",
    views_base: str,
    da3: "DA3Model",
    test_id: int | str = 0,
    dist_thresh: float = 0.2,
    angle_thresh: float = 1,
    step_degrees: int = 20,
    keep_rate_threshold: float = 0.6,
):
    """The GPU-side primitive a client-side graph algorithm calls once per
    candidate edge: run one real pairwise DA3 test between two already-
    downloaded panos, using an already-loaded DA3Model (loading is
    expensive -- callers making many calls in one ZeroGPU session should
    load once and reuse, same as _run_da3's own da3= parameter).

    This is deliberately the ONLY thing this package knows about "an edge"
    -- no notion of dates, corridors, or coverage. Returns None if either
    pano fails the keep-rate health check, else (pose_a, pose_b, pts,
    cols):
      - pose_*: (center: np.ndarray(3,), rotation: np.ndarray(3,3)) --
        world-to-pano rotation, in THIS call's own arbitrary local frame.
        Use rigid_align to stitch onto a caller-side accumulated frame.
      - pts, cols: this edge's own backprojected points/colors, same local
        frame as pose_*.
    """
    test_dir = os.path.join(views_base, f"t{test_id}")
    os.makedirs(test_dir, exist_ok=True)
    id_a, id_b = os.path.basename(path_a), os.path.basename(path_b)
    _, res, pts, cols, _, _ = _run_da3(
        path_a, [path_b], cfg, test_dir,
        da3=da3, dist_thresh=dist_thresh, angle_thresh=angle_thresh, step_degrees=step_degrees,
    )
    ka, ta = res.pano_keep_counts.get(id_a, (0, 1))
    kb, tb = res.pano_keep_counts.get(id_b, (0, 1))
    if (ka / ta) < keep_rate_threshold or (kb / tb) < keep_rate_threshold:
        return None
    pose_a = (res.pano_poses[id_a]["center"], res.pano_poses[id_a]["rotation"])
    pose_b = (res.pano_poses[id_b]["center"], res.pano_poses[id_b]["rotation"])
    return pose_a, pose_b, pts, cols


def score_pano_da3(
    path: str,
    cfg: "PipelineConfig",
    views_base: str,
    da3: "DA3Model",
    score_id: int | str = 0,
    dist_thresh: float = 0.2,
    angle_thresh: float = 1,
    step_degrees: int = 20,
) -> int:
    """The GPU-side primitive a client-side graph algorithm calls once per
    candidate pano to rate it BEFORE spending a real pairwise test on it:
    extract that pano's own view slices and run them through DA3 alone (no
    other pano in the batch), then count how many survive DA3's own
    consensus filter. An internally coherent pano (high count) is more
    likely to pair well with a real neighbor than a low one -- validated
    against real data (see google-map-to-3d's
    tests/debug_solo_score_experiment.py): pairwise success rate rose
    monotonically with the weaker candidate's score, 33% at score 6 up to
    100% at score 13+.

    Same already-loaded DA3Model/views_base as test_edge_da3 -- callers
    doing many of these in one ZeroGPU session should share both rather
    than reload/re-init.
    """
    score_dir = os.path.join(views_base, f"s{score_id}")
    os.makedirs(score_dir, exist_ok=True)
    views = extract_views_for_da3(path, score_dir, prefix=f"s{score_id}_", pano_id=0, step_degrees=step_degrees)
    filtered_views, _ = da3.process_views(views, dist_thresh=dist_thresh, angle_thresh=angle_thresh)
    return len(filtered_views)


def test_edge_da3_bridge(
    path_a: str,
    path_b: str,
    cfg: "PipelineConfig",
    views_base: str,
    da3: "DA3Model",
    test_id: int | str = 0,
    dist_thresh: float = 0.2,
    angle_thresh: float = 1,
    step_degrees: int = 20,
) -> dict | None:
    """Diagnostic variant of test_edge_da3 for a client-side bridging
    search (joining two already-built pieces -- a real DA3 estimate,
    even a poor one, is trusted over independent GPS placement between
    them; GPS is only ever used once, to anchor the final combined
    result to real-world coordinates, not to reconcile pieces against
    each other). Never gates pass/fail itself -- the caller ranks
    several attempts using the raw keep-rate/deviation data returned
    here and always uses the best one found, however weak.

    Returns None only if a pano has no pose at all in this DA3 call --
    an extremely rare case (e.g. view extraction itself produced
    nothing usable), since DA3Model now always provides a pose per pano
    regardless of keep-rate (falling back to the raw pre-filter
    consensus when nothing survives strict filtering -- see
    DA3Model._filter_at_threshold). Else a dict: pose_a/pose_b (center,
    rotation), pts, cols (same as test_edge_da3, empty arrays if
    nothing survived filtering), keep_a/keep_b ((kept, total) view
    counts), and avg_dev_a/avg_dev_b (average real-world deviation in
    meters among that pano's own KEPT views only -- a single wild
    outlier gets filtered out by the keep-rate check anyway, so it says
    nothing about quality; the kept views still not agreeing well with
    each other on average is what actually flags a bad pairing. inf if
    zero views were kept).
    """
    test_dir = os.path.join(views_base, f"b{test_id}")
    os.makedirs(test_dir, exist_ok=True)
    id_a, id_b = os.path.basename(path_a), os.path.basename(path_b)
    _, res, pts, cols, _, _ = _run_da3(
        path_a, [path_b], cfg, test_dir,
        da3=da3, dist_thresh=dist_thresh, angle_thresh=angle_thresh, step_degrees=step_degrees,
    )
    if id_a not in res.pano_poses or id_b not in res.pano_poses:
        return None
    ka, ta = res.pano_keep_counts.get(id_a, (0, 1))
    kb, tb = res.pano_keep_counts.get(id_b, (0, 1))
    pose_a = (res.pano_poses[id_a]["center"], res.pano_poses[id_a]["rotation"])
    pose_b = (res.pano_poses[id_b]["center"], res.pano_poses[id_b]["rotation"])
    return {
        "pose_a": pose_a, "pose_b": pose_b,
        "pts": pts if pts is not None else np.zeros((0, 3)),
        "cols": cols if cols is not None else np.zeros((0, 3)),
        "keep_a": (ka, ta), "keep_b": (kb, tb),
        "avg_dev_a": res.pano_avg_deviation.get(id_a, float("inf")),
        "avg_dev_b": res.pano_avg_deviation.get(id_b, float("inf")),
    }


def save_da3_pointcloud(points: np.ndarray, colors: np.ndarray, path: str) -> str:
    """Thin public wrapper over components.Saver -- for callers outside this
    package that need to save a raw point cloud without reaching into this
    package's internal components.* modules directly."""
    Saver.save_point_cloud(points, path, colors=colors)
    return path


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
    a fresh one -- for a caller making several of these calls in one GPU
    session, which would otherwise reload the model each time. Default
    None preserves the original behavior (load, use, delete) for every
    other caller.

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

