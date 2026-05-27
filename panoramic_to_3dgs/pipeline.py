import os
import json
import tempfile
import contextlib
import torch
import numpy as np
from typing import Optional

from components.SplatGenerator.SplatGenerator import SplatGenerator
from components.DepthMapGenerator.DA360DepthModel import DA360DepthModel
from components.DepthMapGenerator.DA3Model import DA3Model
from components.SplatProcessor.SplatProcessor import SplatProcessor
from components.ViewExtractor.ViewExtractor import extract_views, extract_views_for_da3
from components.Saver.Saver import Saver
from components.SplatProcessor.utils import backproject_views_to_pcd
from sharp.utils.gaussians import Gaussians3D, save_ply

from panoramic_to_3dgs.config import PipelineConfig


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
        panorama_paths: list[str],
        output_dir: str,
        generate_pano_ids: Optional[list[int]] = None,
        external_depth_paths: Optional[list[str]] = None,
    ) -> tuple[Gaussians3D, dict[int, Gaussians3D]]:
        """Run the full pipeline: align, process, and merge Gaussian splats.

        Args:
            panorama_paths: Paths to input panorama images.
            output_dir: Directory to write outputs.
            generate_pano_ids: If set, only generate splats for these pano indices.
                               All panos still contribute to depth estimation.
            external_depth_paths: Optional per-pano depth map paths (depth_mode='external').

        Returns:
            (merged_splat, per_pano_splats)
        """
        cfg = self.config
        debug = cfg.debug
        print(f"Starting pipeline for {len(panorama_paths)} panoramas | Mode: {cfg.depth_mode} | Debug: {debug}")
        os.makedirs(output_dir, exist_ok=True)
        saver = Saver() if debug else None

        all_sharp_views = []
        all_da3_views = []
        pano_poses = None
        da3_pts_per_pano = None

        da360_model = None
        if cfg.depth_mode == "da360":
            da360_model = DA360DepthModel(cfg.da360_model)
            da3_pts_per_pano = {}

        with contextlib.ExitStack() as stack:
            # In debug mode, write view slices into output_dir so they persist.
            # Otherwise use a temp dir that is deleted automatically when the run finishes.
            if debug:
                views_base = output_dir
            else:
                views_base = stack.enter_context(tempfile.TemporaryDirectory())

            for i, pano_path in enumerate(panorama_paths):
                print(f"--- Processing Panorama {i+1}: {pano_path} ---")
                current_image = pano_path
                if cfg.clean_image:
                    from components.ImageCleaner.ImageCleaner import ImageCleaner
                    cleaner = ImageCleaner()
                    cleaned_path = os.path.join(output_dir, f"cleaned_pano_{i}.png")
                    cleaner.clean(current_image, output_path=cleaned_path)
                    current_image = cleaned_path

                if cfg.depth_mode == "da360":
                    pano_depth, pano_rgb = da360_model.predict(current_image)
                    world_pts, world_cols = DA360DepthModel.to_world_pts(pano_depth, pano_rgb)
                    da3_pts_per_pano[i] = world_pts
                    if debug:
                        saver.save_point_cloud(
                            world_pts,
                            os.path.join(output_dir, f"da360_debug_pano_{i}.ply"),
                            colors=world_cols,
                        )
                elif cfg.depth_mode == "external" and external_depth_paths:
                    import cv2
                    pano_depth = cv2.imread(external_depth_paths[i], cv2.IMREAD_UNCHANGED)

                sharp_dir = os.path.join(views_base, f"views_pano_{i}_sharp")
                os.makedirs(sharp_dir, exist_ok=True)
                all_sharp_views.extend(
                    extract_views(
                        current_image,
                        sharp_dir,
                        overlap_degrees=20,
                        slice_count=cfg.slice_count,
                        prefix=f"pano_{i}_",
                        panorama_depth=None,
                        pano_id=i,
                        include_sky=cfg.include_sky,
                    )
                )

                if cfg.depth_mode == "da3":
                    da3_dir = os.path.join(views_base, f"views_pano_{i}_da3")
                    os.makedirs(da3_dir, exist_ok=True)
                    all_da3_views.extend(
                        extract_views_for_da3(
                            current_image, da3_dir, prefix=f"pano_{i}_", pano_id=i
                        )
                    )

            if cfg.depth_mode == "da360":
                if len(panorama_paths) == 1:
                    pano_poses = {0: {"center": np.zeros(3), "rotation": np.eye(3)}}
                else:
                    print(
                        "Warning: multi-pano DA360 has no inter-pano pose estimation; "
                        "depth alignment disabled."
                    )
                    da3_pts_per_pano = None

            if cfg.depth_mode == "da3":
                print("--- Step: DA3 Global Pose Processing ---")
                da3 = DA3Model(cfg.da3_model)
                filtered_da3_views, da3_result = da3.process_views(all_da3_views)
                pano_poses = da3_result.pano_poses

                da3_pts, da3_cols, da3_pts_per_pano = backproject_views_to_pcd(
                    filtered_da3_views, da3_result
                )
                if debug and da3_pts is not None:
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

                del da3, da3_result, filtered_da3_views, da3_cols, da3_pts
                torch.cuda.empty_cache()

            if generate_pano_ids is not None:
                generate_set = set(generate_pano_ids)
                all_sharp_views = [v for v in all_sharp_views if v.pano_id in generate_set]
            print(
                f"Generating splats for {len(all_sharp_views)} views across panos: "
                f"{sorted({v.pano_id for v in all_sharp_views})}"
            )

            print("--- Step: Splat Generation (SHARP) ---")
            gs_generator = SplatGenerator(cfg.sharp_model)
            splat_out_dir = os.path.join(output_dir, "gs") if debug else None
            gaussian_list = gs_generator.generate_from_views(all_sharp_views, output_dir=splat_out_dir)

            # ExitStack closes here — temp dirs deleted after SHARP reads view slices
            # but before we write final PLYs (which go to output_dir, not views_base).

        print("--- Step: Splat Processing (Alignment/Merge) ---")
        processor = SplatProcessor(
            num_z_slabs=cfg.num_z_slabs,
            num_fov_slabs=cfg.num_fov_slabs,
            smooth_sigma_m=cfg.smooth_sigma_m,
            smooth_sigma_fov=cfg.smooth_sigma_fov,
            voronoi_buffer_m=cfg.voronoi_buffer_m,
            floor_keep_fraction=cfg.floor_keep_fraction,
            min_depth_coverage=cfg.min_depth_coverage,
            align_depth=cfg.align_depth,
            near_depth=cfg.near_depth,
            sky_depth=cfg.sky_depth,
        )
        merged_splat, per_pano_splats = processor.process(
            all_sharp_views,
            gaussian_list,
            pano_poses=pano_poses,
            da3_world_pts=da3_pts_per_pano,
            scale_mode=cfg.scale_mode,
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

        if debug:
            for pid, splat in per_pano_splats.items():
                pano_out = os.path.join(output_dir, f"output_pano_{pid}.ply")
                save_ply(
                    splat,
                    f_px=ref_view.focal_px,
                    image_shape=(ref_view.height, ref_view.width),
                    path=pano_out,
                )
                print(f"Saved pano {pid}: {pano_out}")

        return merged_splat, per_pano_splats
