import os
import torch
import cv2
import numpy as np
from components.SplatGenerator.SplatGenerator import SplatGenerator
from components.DepthMapGenerator.DA360DepthModel import DA360DepthModel
from components.DepthMapGenerator.DA3Model import DA3Model
from components.SplatProcessor.SplatProcessor import SplatProcessor
from components.ImageCleaner.ImageCleaner import ImageCleaner
from components.ViewExtractor.ViewExtractor import extract_views
from components.Saver.Saver import Saver
from sharp.utils.gaussians import save_ply

from components.SplatProcessor.utils import panoramic_depth_to_pcd

def run_panoramic_pipeline(
    panorama_path,
    output_dir,
    clean_image=False,
    depth_mode=None, # 'da360' or 'da3' or 'external' or None
    external_depth_path=None,
    model_paths=None
):
    """
    Standard pipeline: panoramic -> (clean) -> extract -> (depth) -> splat -> process
    """
    print(f"Starting pipeline for: {panorama_path} | Mode: {depth_mode}")
    os.makedirs(output_dir, exist_ok=True)
    saver = Saver()
    
    # 1. Clean Panorama (Optional)
    current_image = panorama_path
    if clean_image:
        print("--- Step: Image Cleaning ---")
        cleaner = ImageCleaner()
        cleaned_path = os.path.join(output_dir, "cleaned_panorama.png")
        cleaner.clean(current_image, output_path=cleaned_path)
        current_image = cleaned_path

    # 2. Panorama-level Depth
    panorama_depth = None
    if depth_mode == 'da360':
        print("--- Step: DA360 Depth Generation ---")
        da360 = DA360DepthModel(model_paths['da360'])
        panorama_depth, panorama_image_rgb = da360.predict(current_image)
    
    elif depth_mode == 'external' and external_depth_path:
        print(f"--- Step: Loading External Depth from {external_depth_path} ---")
        # Load as-is (supporting uint16/LiDAR depth maps)
        panorama_depth = cv2.imread(external_depth_path, cv2.IMREAD_UNCHANGED)
        if panorama_depth is None:
            raise ValueError(f"Could not load external depth at {external_depth_path}")
        # Load image for coloring
        panorama_image_rgb = cv2.cvtColor(cv2.imread(current_image), cv2.COLOR_BGR2RGB)

    # Save simple visual results for any panorama-level depth
    if panorama_depth is not None:
        saver.save_depth_image(panorama_depth, os.path.join(output_dir, "panorama_depth_visual.jpg"))
        try:
            # Use the new utility to generate filtered PCD
            pcd_points, pcd_colors = panoramic_depth_to_pcd(panorama_depth, panorama_image_rgb)
            saver.save_point_cloud(pcd_points, os.path.join(output_dir, "panorama_depth_debug.ply"), colors=pcd_colors)
        except Exception as e:
            print(f"Warning: Could not save debug point cloud: {e}")

    # 3. Extract Views
    print("--- Step: View Extraction ---")
    views_output_dir = os.path.join(output_dir, "views")
    os.makedirs(views_output_dir, exist_ok=True)
    views_data = extract_views(
        current_image,
        views_output_dir,
        overlap_degrees=9,
        slice_count=4,
        panorama_depth=panorama_depth 
    )

    # 4. View-level Depth (DA3)
    if depth_mode == 'da3':
        print("--- Step: DA3 Multi-view Depth/Pose Generation ---")
        da3 = DA3Model(model_paths['da3'])
        da3.process_views(views_data) 

    # 5. Generate Splats
    print("--- Step: Splat Generation (SHARP) ---")
    gs_generator = SplatGenerator(model_paths['sharp'])
    gs_output_dir = os.path.join(output_dir, "gs")
    gaussian_list = gs_generator.generate_from_views(views_data, output_dir=gs_output_dir)

    # 6. Process and Merge
    print("--- Step: Splat Processing (Alignment/Merge) ---")
    processor = SplatProcessor()
    merged_splat = processor.process(views_data, gaussian_list, enable_alignment=(depth_mode is not None))
    
    # 7. Save Final Result
    final_path = os.path.join(output_dir, "final_output.ply")
    save_ply(
        merged_splat, 
        f_px=views_data[0].focal_px, 
        image_shape=(views_data[0].height, views_data[0].width), 
        path=final_path
    )
    print(f"Pipeline complete. Final result saved to {final_path}")

if __name__ == '__main__':
    # Paths & Configuration
    models = {
        'da360': "models/DA360_large.pth",
        'da3': "models/models--depth-anything--DA3NESTED-GIANT-LARGE-1.1/snapshots/b2359bdf726fb44ef62acca04d629dcf158053e7",
        'sharp': "models/sharp_2572gikvuh.pt"
    }

    run_panoramic_pipeline(
        panorama_path='data/inputs/round1.jpg',
        output_dir='data/outputs/modular_run',
        clean_image=False,
        depth_mode=None, # 'da360' or 'da3' or 'external' or None
        # external_depth_path='data_helvipad/2024/0001.png',
        model_paths=models
    )
