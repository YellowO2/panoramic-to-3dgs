import os
from functions.sharp_infer import extracted_views_to_3dgs
from functions.process_splats import process_splats
from functions.extract_views_from_panorama import extract_views, extract_views_for_depthmap
from functions.depth_align import get_da360_panorama_depth
from sharp.utils.gaussians import save_ply




if __name__ == '__main__':
    output_dir = 'data/output_views'
    os.makedirs(output_dir, exist_ok=True) 
    
    panoramas = [
        # 'round1.jpg',
        # 'round2.jpg',
        'data/inputs/cleaned_test_output.png',
        # add more panoramas here
    ]
    
    # toggle between DA3 and DA360 modes
    use_da360 = True 

    views_data = []
    for i, pano_image in enumerate(panoramas):
        
        panorama_depth = None
        if use_da360:
            print(f"generating depthmap using DA360 for {pano_image}...")
            panorama_depth = get_da360_panorama_depth(pano_image, model_path="models/DA360_large.pth", save_debug_ply="data/outputs/debug_da360.ply")
            
        views_data = extract_views(
            pano_image,
            output_dir,
            overlap_degrees=9,
            slice_count=4,
            prefix=f"panorama_{i}_",
            panorama_depth=panorama_depth
        )
        views_data.extend(views_data)
        
    print("finish 1")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = 'data/output_3dgs'
    os.makedirs(output_dir, exist_ok=True) 
    model_path = os.path.join(script_dir, "models", "sharp_2572gikvuh.pt")
    
    test_views = views_data[:1] 
    
    # === SHARP generation of each view ===
    gaussian_list = extracted_views_to_3dgs(
        views_data,
        model_path=model_path,
        output_dir=output_dir,
    )

    # === process splats generated ===
    # process the splats without depth
    # merged_splat_unaligned = process_splats(depthmap_views_data, gaussian_list, enable_alignment=False)
    # save_ply(
    #     merged_splat_unaligned, 
    #     f_px=test_views[0]["focal_px"], 
    #     image_shape=(test_views[0]["height"], test_views[0]["width"]), 
    #     path=os.path.join(output_dir, "final_unaligned.ply")
    # )
    # print(f"Saved unaligned splat to {unaligned_path}")

    # process splats with depth
    merged_splat_aligned = process_splats(views_data, gaussian_list, enable_alignment=True)
    aligned_path = os.path.join(output_dir, "final_aligned.ply")
    save_ply(
        merged_splat_aligned, 
        f_px=test_views[0]["focal_px"], 
        image_shape=(test_views[0]["height"], test_views[0]["width"]), 
        path=aligned_path
    )
    print(f"Saved DA3-aligned splat to {aligned_path}")

