import os
import json
import torch
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from components.SplatGenerator.SplatGenerator import SplatGenerator
from components.DepthMapGenerator.DA360DepthModel import DA360DepthModel
from components.DepthMapGenerator.DA3Model import DA3Model
from components.SplatProcessor.SplatProcessor import SplatProcessor
from components.ImageCleaner.ImageCleaner import ImageCleaner
from components.ViewExtractor.ViewExtractor import extract_views, extract_views_for_da3
from components.Saver.Saver import Saver
from sharp.utils.gaussians import save_ply
from components.SplatProcessor.utils import (
    panoramic_depth_to_pcd,
    backproject_views_to_pcd,
)


_EARTH_R = 6_371_000.0  # metres


def load_panorama_folder(folder_path: str) -> tuple[list[str], list[str | None], list[dict]]:
    """Load panoramas from a folder containing metadata.json and pano_{id}.jpg / pano_{id}_depth.npy files.

    Returns (panorama_paths, depth_paths, metadata) where depth_paths[i] is None if no depth file exists.
    """
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


def compute_google_pano_poses(metadata: list[dict]) -> dict[int, dict]:
    """Build pano_poses from Google metadata (heading/lat/lon in radians/degrees).

    The first panorama is the ENU origin (center = [0, 0, 0]).
    Returns pano_poses dict keyed by pano index (0, 1, ...).
    heading and pitch are expected in radians.
    """
    ref_lat = np.radians(metadata[0]["lat"])
    ref_lon = np.radians(metadata[0]["lon"])

    pano_poses = {}
    for i, entry in enumerate(metadata):
        lat = np.radians(entry["lat"])
        lon = np.radians(entry["lon"])
        h = entry["heading"]  # radians from North, clockwise

        # ENU position relative to first panorama
        east  = (lon - ref_lon) * np.cos(ref_lat) * _EARTH_R
        north = (lat - ref_lat) * _EARTH_R
        center = np.array([east, north, 0.0])

        # Camera frame: X=right, Y=up, Z=forward (into scene)
        # R_c2w columns = [right_in_ENU, up_in_ENU, forward_in_ENU]
        R_c2w = np.array([
            [ np.cos(h),  0.0,  np.sin(h)],   # East row
            [-np.sin(h),  0.0,  np.cos(h)],   # North row
            [ 0.0,        1.0,  0.0       ],   # Up row
        ])
        pano_rot = R_c2w.T  # R_w2pano — convention expected by SplatProcessor

        pano_poses[i] = {"center": center, "rotation": pano_rot}
        print(f"  [Google pose] Pano {i}: center=({east:.2f}m E, {north:.2f}m N), heading={np.degrees(h):.1f}°")

    return pano_poses


def load_google_depth_pts(
    metadata: list[dict],
    folder_path: str,
    pano_poses: dict[int, dict],
) -> dict[int, np.ndarray]:
    """Backproject Google equirectangular depth maps to world-space point clouds.

    Returns da3_pts_per_pano dict keyed by pano index, same format as DA3 output.
    Sky pixels (depth <= 0) are discarded.
    """
    da3_pts_per_pano = {}
    for i, entry in enumerate(metadata):
        if not entry.get("has_depth"):
            continue
        depth_path = os.path.join(folder_path, f"pano_{entry['id']}_depth.npy")
        if not os.path.exists(depth_path):
            print(f"  [Google depth] Pano {i}: depth file missing, skipping.")
            continue

        depth = np.load(depth_path).astype(np.float32)
        # panoramic_depth_to_pcd filters d <= 1e-3 (handles -1 sky sentinel)
        local_pts, _ = panoramic_depth_to_pcd(depth)
        if local_pts is None or len(local_pts) == 0:
            continue

        pose = pano_poses[i]
        R_c2w = pose["rotation"].T  # pano_rot.T = R_c2w
        center = pose["center"]
        world_pts = (R_c2w @ local_pts.T).T + center
        da3_pts_per_pano[i] = world_pts
        print(f"  [Google depth] Pano {i}: {len(world_pts)} world points.")

    return da3_pts_per_pano


def run_panoramic_pipeline(
    panorama_paths: list[str],
    output_dir: str,
    clean_image=False,
    depth_mode=None,  # 'da360' | 'da3' | 'google' | 'external' | None
    external_depth_paths: list[str] = None,
    metadata: list[dict] = None,       # required for depth_mode='google'
    folder_path: str = None,           # required for depth_mode='google'
    model_paths=None,
):
    print(f"Starting pipeline for {len(panorama_paths)} panoramas | Mode: {depth_mode}")
    os.makedirs(output_dir, exist_ok=True)
    saver = Saver()

    all_sharp_views = []
    all_da3_views = []

    for i, pano_path in enumerate(panorama_paths):
        print(f"--- Processing Panorama {i+1}: {pano_path} ---")
        current_image = pano_path
        if clean_image:
            cleaner = ImageCleaner()
            cleaned_path = os.path.join(output_dir, f"cleaned_pano_{i}.png")
            cleaner.clean(current_image, output_path=cleaned_path)
            current_image = cleaned_path

        # individual DA360 for alignment/debug if requested
        pano_depth = None
        # if depth_mode == 'da360':
        #     da360 = DA360DepthModel(model_paths['da360'])
        #     pano_depth, pano_rgb = da360.predict(current_image)
        #     pcd_pts, pcd_cols = panoramic_depth_to_pcd(pano_depth, pano_rgb)
        #     saver.save_point_cloud(pcd_pts, os.path.join(output_dir, f"pano_{i}_da360.ply"), colors=pcd_cols)
        # elif depth_mode == 'external' and external_depth_paths:
        #     pano_depth = cv2.imread(external_depth_paths[i], cv2.IMREAD_UNCHANGED)

        # A. Extract SHARP views (for splats)
        sharp_dir = os.path.join(output_dir, f"views_pano_{i}_sharp")
        os.makedirs(sharp_dir, exist_ok=True)
        all_sharp_views.extend(
            extract_views(
                current_image,
                sharp_dir,
                overlap_degrees=10,
                slice_count=6,
                prefix=f"pano_{i}_",
                panorama_depth=pano_depth,
                pano_id=i,
            )
        )

        # B. Extract DA3 views (for global poses) — skipped in google mode
        if depth_mode == "da3":
            da3_dir = os.path.join(output_dir, f"views_pano_{i}_da3")
            os.makedirs(da3_dir, exist_ok=True)
            all_da3_views.extend(
                extract_views_for_da3(
                    current_image, da3_dir, prefix=f"pano_{i}_", pano_id=i
                )
            )

    # 4. Global Multi-View Depth/Pose Generation (DA3)
    pano_poses = None
    da3_pts_per_pano = None
    if depth_mode == "da3":
        print("--- Step: DA3 Global Pose Processing ---")
        da3 = DA3Model(model_paths["da3"])
        # DA3 uses the optimized 16:9 slices and returns only "good" ones
        filtered_da3_views, da3_result = da3.process_views(all_da3_views)
        pano_poses = da3_result.pano_poses

        # Save DA3 Debug PCD (Verifies the cleaned scene)
        print("--- Step: Saving DA3 Debug Consistency PCD ---")
        da3_pts, da3_cols, da3_pts_per_pano = backproject_views_to_pcd(
            filtered_da3_views, da3_result
        )
        if da3_pts is not None:
            saver.save_point_cloud(
                da3_pts,
                os.path.join(output_dir, "da3_debug_consistency.ply"),
                colors=da3_cols,
            )
            for pid, pts in da3_pts_per_pano.items():
                saver.save_point_cloud(
                    pts, os.path.join(output_dir, f"da3_debug_pano_{pid}.ply")
                )

        del da3, da3_result, filtered_da3_views, da3_cols, da3_pts
        torch.cuda.empty_cache()

    elif depth_mode == "google":
        print("--- Step: Google Metadata Pose + Depth Processing ---")
        pano_poses = compute_google_pano_poses(metadata)
        da3_pts_per_pano = load_google_depth_pts(metadata, folder_path, pano_poses)
        for pid, pts in da3_pts_per_pano.items():
            saver.save_point_cloud(
                pts, os.path.join(output_dir, f"google_debug_pano_{pid}.ply")
            )

    # 5. Generate Splats (SHARP)
    print("--- Step: Splat Generation (SHARP) ---")
    gs_generator = SplatGenerator(model_paths["sharp"])
    all_sharp_views = all_sharp_views[:]  # use less slice now as not enough ram
    print(f"length of all_sharp_views: {len(all_sharp_views)}")
    gaussian_list = gs_generator.generate_from_views(
        all_sharp_views, output_dir=os.path.join(output_dir, "gs")
    )

    # 6. Process and Merge
    print("--- Step: Splat Processing (Alignment/Merge) ---")
    processor = SplatProcessor()
    merged_splat, per_pano_splats = processor.process(
        all_sharp_views,
        gaussian_list,
        pano_poses=pano_poses,
        da3_world_pts=da3_pts_per_pano,
        scale_mode="da3_2dgrid_global",
    )

    # 7. Save Final Result
    ref_view = all_sharp_views[0]
    final_path = os.path.join(output_dir, "final_output.ply")
    save_ply(
        merged_splat,
        f_px=ref_view.focal_px,
        image_shape=(ref_view.height, ref_view.width),
        path=final_path,
    )
    print(f"Pipeline complete: {final_path}")

    for pid, splat in per_pano_splats.items():
        pano_path = os.path.join(output_dir, f"output_pano_{pid}.ply")
        save_ply(
            splat,
            f_px=ref_view.focal_px,
            image_shape=(ref_view.height, ref_view.width),
            path=pano_path,
        )
        print(f"Saved pano {pid}: {pano_path}")


if __name__ == "__main__":
    models = {
        "da360": "models/DA360_large.pth",
        "da3": "models/models--depth-anything--DA3NESTED-GIANT-LARGE-1.1/snapshots/b2359bdf726fb44ef62acca04d629dcf158053e7",
        "sharp": "models/sharp_2572gikvuh.pt",
    }

    # --- Option A: folder input with Google metadata poses ---
    folder = "data/inputs/panoramas_example"
    panos, depths, meta = load_panorama_folder(folder)
    run_panoramic_pipeline(
        panorama_paths=panos,
        output_dir="data/outputs/folder_test",
        depth_mode="google",
        metadata=meta,
        folder_path=folder,
        model_paths=models,
    )

    # --- Option B: manual list (legacy) ---
    # panos = [
    #     "data/inputs/round1.jpg",
    #     "data/inputs/round2.jpg",
    #     "data/inputs/round3_2.jpg",
    #     "data/inputs/round_4.jpg",
    # ]
    # run_panoramic_pipeline(
    #     panorama_paths=panos,
    #     output_dir="data/outputs/multi_pano_test",
    #     depth_mode="da3",
    #     model_paths=models,
    # )
