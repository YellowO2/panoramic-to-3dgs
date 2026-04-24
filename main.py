import os
import torch
from components.SplatGenerator.SplatGenerator import SplatGenerator
from components.DepthMapGenerator.DA360DepthModel import DA360DepthModel
from components.DepthMapGenerator.DA3Model import DA3Model
from components.SplatProcessor.SplatProcessor import SplatProcessor
from components.ImageCleaner.ImageCleaner import ImageCleaner
from components.ViewExtractor.ViewExtractor import extract_views
from components.Saver.Saver import Saver
from sharp.utils.gaussians import save_ply

def run_panoramic_pipeline(
    panorama_path,
    output_dir,
    clean_image=False,
    depth_mode='da360', # 'da360' or 'da3' or None
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

    # 2. Panorama-level Depth (DA360)
    panorama_depth = None
    if depth_mode == 'da360':
        print("--- Step: DA360 Depth Generation ---")
        da360 = DA360DepthModel(model_paths['da360'])
        panorama_depth, _ = da360.predict(current_image)
        
        # Save simple visual results
        saver.save_depth_image(panorama_depth, os.path.join(output_dir, "da360_depth.jpg"))
        saver.save_point_cloud(panorama_depth, current_image, os.path.join(output_dir, "da360_debug.ply"))

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
        panorama_path='data/inputs/cleaned_test_output.png',
        output_dir='data/outputs/modular_run',
        clean_image=False,
        depth_mode='da360',
        model_paths=models
    )
