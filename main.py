import os
import torch
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from components.SplatGenerator.SplatGenerator import SplatGenerator
from components.DepthMapGenerator.DA360DepthModel import DA360DepthModel
from components.DepthMapGenerator.DA3Model import DA3Model
from components.SplatProcessor.SplatProcessor import SplatProcessor
from components.ImageCleaner.ImageCleaner import ImageCleaner
from components.ViewExtractor.ViewExtractor import extract_views
from components.Saver.Saver import Saver
from sharp.utils.gaussians import save_ply
from components.SplatProcessor.utils import panoramic_depth_to_pcd, backproject_views_to_pcd

def run_panoramic_pipeline(
    panorama_paths: list[str],
    output_dir: str,
    clean_image=False,
    depth_mode=None, # 'da360' or 'da3' or 'external' or None
    external_depth_paths: list[str] = None,
    model_paths=None
):
    print(f"Starting pipeline for {len(panorama_paths)} panoramas | Mode: {depth_mode}")
    os.makedirs(output_dir, exist_ok=True)
    saver, all_views_data = Saver(), []
    
    for i, pano_path in enumerate(panorama_paths):
        print(f"--- Processing Panorama {i+1}: {pano_path} ---")
        current_image = pano_path
        if clean_image:
            cleaner = ImageCleaner()
            cleaned_path = os.path.join(output_dir, f"cleaned_pano_{i}.png")
            cleaner.clean(current_image, output_path=cleaned_path)
            current_image = cleaned_path

        # Generate DA360 Depth for individual debug
        pano_depth = None
        if depth_mode == 'da360':
            da360 = DA360DepthModel(model_paths['da360'])
            pano_depth, pano_rgb = da360.predict(current_image)
            pcd_pts, pcd_cols = panoramic_depth_to_pcd(pano_depth, pano_rgb)
            saver.save_point_cloud(pcd_pts, os.path.join(output_dir, f"pano_{i}_da360.ply"), colors=pcd_cols)
        elif depth_mode == 'external' and external_depth_paths:
            pano_depth = cv2.imread(external_depth_paths[i], cv2.IMREAD_UNCHANGED)

        # View Extraction
        pano_views_dir = os.path.join(output_dir, f"views_pano_{i}")
        os.makedirs(pano_views_dir, exist_ok=True)
        views_data = extract_views(current_image, pano_views_dir, overlap_degrees=9, slice_count=4, prefix=f"pano_{i}_", panorama_depth=pano_depth, pano_id=i)
        all_views_data.extend(views_data)

    # 4. Global Multi-View Depth/Pose Generation (DA3)
    pano_poses = None
    if depth_mode == 'da3':
        print("--- Step: DA3 Global Processing ---")
        da3 = DA3Model(model_paths['da3'])
        horizon_views = [v for v in all_views_data if abs(v.pitch) < 1e-3]
        da3_result = da3.process_views(horizon_views) 
        pano_poses = da3_result.pano_poses

        # Save DA3 Debug Consistency Point Cloud (Only horizon slices have DA3 results)
        print("--- Step: Saving DA3 Debug Consistency PCD ---")
        da3_pts, da3_cols = backproject_views_to_pcd(horizon_views, da3_result)
        if da3_pts is not None:
            saver.save_point_cloud(da3_pts, os.path.join(output_dir, "da3_debug_consistency.ply"), colors=da3_cols)

    # 5. Generate Splats (SHARP)
    print("--- Step: Splat Generation (SHARP) ---")
    gaussian_list = SplatGenerator(model_paths['sharp']).generate_from_views(all_views_data, output_dir=os.path.join(output_dir, "gs"))

    # 6. Process and Merge
    print("--- Step: Splat Processing (Alignment/Merge) ---")
    merged_splat = SplatProcessor().process(all_views_data, gaussian_list, pano_poses=pano_poses)
    
    # 7. Save Final Result
    final_path = os.path.join(output_dir, "final_output.ply")
    save_ply(merged_splat, f_px=all_views_data[0].focal_px, image_shape=(all_views_data[0].height, all_views_data[0].width), path=final_path)
    print(f"Pipeline complete: {final_path}")

if __name__ == '__main__':
    models = {'da360': "models/DA360_large.pth", 'da3': "models/models--depth-anything--DA3NESTED-GIANT-LARGE-1.1/snapshots/b2359bdf726fb44ef62acca04d629dcf158053e7", 'sharp': "models/sharp_2572gikvuh.pt"}
    panos = ['data/inputs/round1.jpg', 'data/inputs/round2.jpg', 'data/inputs/round3.jpg']
    run_panoramic_pipeline(panorama_paths=panos, output_dir='data/outputs/multi_pano_test', depth_mode='da3', model_paths=models)
