import os
import tempfile

import numpy as np
import torch
from sharp.utils.gaussians import Gaussians3D, save_ply

from panoramic_to_3dgs.components.SplatGenerator.SplatGenerator import SplatGenerator
from panoramic_to_3dgs.components.SplatProcessor.SplatProcessor import SplatProcessor
from panoramic_to_3dgs.components.ViewExtractor.ViewExtractor import extract_views
from panoramic_to_3dgs.config import PipelineConfig


class Pipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def run(self, target_appearance_path: str, output_dir: str, depth: dict | None = None) -> Gaussians3D:
        """One panorama to one Gaussian splat, scaled against depth made elsewhere.

        Args:
            target_appearance_path: the panorama SHARP builds the splat from.
                May be an edited version (e.g. relit) of the one depth came from.
            output_dir: where final_output.ply is written.
            depth: DA3's view of the scene around this panorama, as
                streetview_to_3d's da3_ops.depth_around returns it --
                points (N, 3) and pose (center, rotation), both in one frame,
                and n_clean, how many DA3 views survived its consensus
                filter. None, or too few clean views, aligns the slices to
                each other only (see SplatProcessor's SHARP-only fallback).

        Returns the merged splat, anchored so the panorama's capture point
        lands at (0, 0, 0) (also saved as final_output.ply).
        """
        cfg = self.config
        os.makedirs(output_dir, exist_ok=True)
        pose = depth.get("pose") if depth else None
        pano_poses = {0: {"center": np.asarray(pose[0]), "rotation": np.asarray(pose[1])}} if pose else {}
        points = depth["points"] if depth and len(depth["points"]) else None
        n_clean = depth["n_clean"] if depth else 0

        with tempfile.TemporaryDirectory() as tmp:
            views_dir = os.path.join(output_dir, "views") if cfg.debug else tmp
            os.makedirs(views_dir, exist_ok=True)
            views = extract_views(target_appearance_path, views_dir, overlap_degrees=20,
                                  slice_count=cfg.slice_count, prefix="pano_0_", pano_id=0,
                                  include_sky=cfg.include_sky)
            print(f"--- SHARP: {len(views)} views of the target pano ---")
            generator = SplatGenerator(cfg.sharp_model)
            splats = generator.generate_from_views(
                views, output_dir=os.path.join(output_dir, "gs") if cfg.debug else None)
            del generator
            torch.cuda.empty_cache()

        print("--- Alignment and merge ---")
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
        merged = processor.process(views, splats, pano_poses=pano_poses, all_da3_pts=points,
                                   scale_mode=cfg.scale_mode, n_da3_clean=n_clean)

        final_path = os.path.join(output_dir, "final_output.ply")
        save_ply(merged, f_px=views[0].focal_px, image_shape=(views[0].height, views[0].width),
                 path=final_path)
        print(f"Pipeline complete: {final_path}")
        del splats, views, processor
        torch.cuda.empty_cache()
        return merged
