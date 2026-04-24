import os
import torch
from components.SplatGenerator.GS3DGenerator import GS3DGenerator
from components.DepthMapGenerator.DA360DepthModel import DA360DepthModel
from components.ImageCleaner.ImageCleaner import ImageCleaner
from functions.extract_views_from_panorama import extract_views
from functions.process_splats import process_splats
from sharp.utils.gaussians import save_ply

def run_panoramic_pipeline(
    panorama_path,
    output_dir,
    clean_image=False,
    use_da360=True,
    model_paths=None
):
    """
    Standard pipeline: panoramic -> (clean) -> (depth) -> slice -> 3dgs -> align -> combine
    """
    print(f"Starting panoramic pipeline for: {panorama_path}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Clean Panorama (Optional)
    current_image = panorama_path
    if clean_image:
        print("--- Step: Image Cleaning ---")
        cleaner = ImageCleaner()
        cleaned_path = os.path.join(output_dir, "cleaned_panorama.png")
        cleaner.clean(current_image, output_path=cleaned_path)
        current_image = cleaned_path

    # 2. Generate Depth Map (Optional)
    panorama_depth = None
    if use_da360:
        print("--- Step: DA360 Depth Generation ---")
        depth_model = DA360DepthModel(model_paths['da360'])
        panorama_depth = depth_model.predict(current_image)
        depth_model.save_debug_ply(panorama_depth, current_image, os.path.join(output_dir, "debug_da360.ply"))

    # 3. Slice Panorama
    print("--- Step: Slicing Panorama ---")
    views_output_dir = os.path.join(output_dir, "views")
    os.makedirs(views_output_dir, exist_ok=True)
    views_data = extract_views(
        current_image,
        views_output_dir,
        overlap_degrees=9,
        slice_count=4,
        panorama_depth=panorama_depth
    )

    # 4. Generate 3DGS
    print("--- Step: 3DGS Generation (SHARP) ---")
    gs_generator = GS3DGenerator(model_paths['sharp'])
    gs_output_dir = os.path.join(output_dir, "gs")
    gaussian_list = gs_generator.generate_from_views(views_data, output_dir=gs_output_dir)

    # 5. Align and Merge
    print("--- Step: Aligning and Merging Splats ---")
    merged_splat = process_splats(views_data, gaussian_list, enable_alignment=use_da360)
    
    # 6. Save Final Result
    final_path = os.path.join(output_dir, "final_aligned.ply")
    save_ply(
        merged_splat, 
        f_px=views_data[0].focal_px, 
        image_shape=(views_data[0].height, views_data[0].width), 
        path=final_path
    )
    print(f"Pipeline complete. Final result saved to {final_path}")

if __name__ == '__main__':
    # Configuration
    panorama_to_process = 'data/inputs/cleaned_test_output.png'
    output_root = 'data/outputs/test_pipeline'
    
    models = {
        'da360': "models/DA360_large.pth",
        'sharp': "models/sharp_2572gikvuh.pt"
    }

    # You can easily toggle steps here
    run_panoramic_pipeline(
        panorama_path=panorama_to_process,
        output_dir=output_root,
        clean_image=False,
        use_da360=True,    # set to False to skip DA360 depth
        model_paths=models
    )
