import os
from functions.sharp_infer import extracted_views_to_3dgs
from functions.process_splats import process_splats
from functions.extract_views_from_panorama import extract_views, extract_views_for_depthmap
from sharp.utils.gaussians import save_ply




if __name__ == '__main__':
    output_dir = 'output_views'
    os.makedirs(output_dir, exist_ok=True) 
    
    panoramas = [
        'panorama_1.347145_103.6917918.jpg',
        # add more panoramas here
    ]
    
    depthmap_views_data = []
    for i, pano_image in enumerate(panoramas):
        views_data = extract_views_for_depthmap(
            pano_image,
            output_dir,
            slice_count=8,
            overlap_degrees=45,
            prefix=f"panorama_{i}_"
        )
        depthmap_views_data.extend(views_data)
        
    print("finish 1")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = 'output_3dgs'
    os.makedirs(output_dir, exist_ok=True) 
    model_path = os.path.join(script_dir, "models", "sharp_2572gikvuh.pt")
    
    # We will use the dense 8-slice extraction for EVERYTHING now (both DA3 and SHARP)
    test_views = depthmap_views_data[:1] 
    
    # 1. SHARP generation on the 8 slices
    gaussian_list = extracted_views_to_3dgs(
        depthmap_views_data,
        model_path=model_path,
        output_dir=output_dir,
    )

    # 2. Merge them WITHOUT Depth Anything alignment
    merged_splat_unaligned = process_splats(depthmap_views_data, gaussian_list, enable_alignment=False)
    unaligned_path = os.path.join(output_dir, "final_unaligned.ply")
    save_ply(
        merged_splat_unaligned, 
        f_px=test_views[0]["focal_px"], 
        image_shape=(test_views[0]["height"], test_views[0]["width"]), 
        path=unaligned_path
    )
    print(f"Saved unaligned splat to {unaligned_path}")

    # 3. Merge them WITH Depth Anything alignment
    merged_splat_aligned = process_splats(depthmap_views_data, gaussian_list, enable_alignment=True)
    aligned_path = os.path.join(output_dir, "final_aligned.ply")
    save_ply(
        merged_splat_aligned, 
        f_px=test_views[0]["focal_px"], 
        image_shape=(test_views[0]["height"], test_views[0]["width"]), 
        path=aligned_path
    )
    print(f"Saved DA3-aligned splat to {aligned_path}")

