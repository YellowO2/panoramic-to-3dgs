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
        target_pano_id: int = 0,
    ) -> Gaussians3D:
        """Run the full pipeline: align, process, and merge Gaussian splats.

        Args:
            panorama_paths: Paths to input panorama images. The target pano plus any
                            nearby supporting panos (used by DA3 for joint depth/pose).
            output_dir: Directory to write outputs.
            target_pano_id: Index of the pano to generate splats for. All other panos
                            are used only for DA3 depth/pose support. The output PLY is
                            anchored so this pano's capture point lands at (0,0,0).

        Returns:
            Merged Gaussian splat (also saved as final_output.ply).
        """
        cfg = self.config
        debug = cfg.debug
        print(f"Starting pipeline for {len(panorama_paths)} panoramas | Debug: {debug}")
        os.makedirs(output_dir, exist_ok=True)
        saver = Saver() if debug else None

        all_sharp_views = []
        all_da3_views = []

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

                da3_dir = os.path.join(views_base, f"views_pano_{i}_da3")
                os.makedirs(da3_dir, exist_ok=True)
                all_da3_views.extend(
                    extract_views_for_da3(
                        current_image, da3_dir, prefix=f"pano_{i}_", pano_id=i
                    )
                )

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

            n_da3_clean = len(filtered_da3_views)
            del da3, da3_result, filtered_da3_views, da3_cols, da3_pts
            torch.cuda.empty_cache()

            all_sharp_views = [v for v in all_sharp_views if v.pano_id == target_pano_id]
            print(
                f"Generating splats for {len(all_sharp_views)} views of target pano {target_pano_id}"
            )

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
            target_pano_id=target_pano_id,
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

        del gaussian_list, all_sharp_views, all_da3_views, processor
        torch.cuda.empty_cache()
        return merged_splat
