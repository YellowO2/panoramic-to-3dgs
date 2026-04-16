import os
from functions.sharp_infer import extracted_views_to_3dgs
from functions.process_splats import process_splats
from functions.extract_views_from_panorama import extract_views, extract_views_for_depthmap
from sharp.utils.gaussians import save_ply




if __name__ == '__main__':
    output_dir = 'output_views'
    os.makedirs(output_dir, exist_ok=True) 
    views_data = extract_views_for_depthmap(
        'panorama_1.347145_103.6917918.jpg',
        output_dir,
        slice_count=12,
        overlap_degrees=30.0,
    )
    print("finish 1")
    
    # output_dir = 'output_views_2'
    # os.makedirs(output_dir, exist_ok=True) 
    # views_data = extract_views_for_depthmap(
    #     'panorama_1.347145_103.6917918.jpg',
    #     output_dir,
    #     slice_count=4,
    #     overlap_degrees=30.0,
    # )
    # print("finish 2")

    # views_data = extract_views(
    #     'panorama_1.347145_103.6917918.jpg',
    #     output_dir,
    #     slice_count=4,
    #     overlap_degrees=20.0,
    # )

    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # output_dir = 'output_3dgs'
    # model_path = os.path.join(script_dir, "models", "sharp_2572gikvuh.pt")
    # test_views = views_data[:1] 
    # gaussian_list = extracted_views_to_3dgs(
    #     views_data,
    #     model_path=model_path,
    #     output_dir=output_dir,
    #     # can specify device param if what to force CPU when u have GPU.
    # )

    # merged_splat =process_splats(views_data, gaussian_list)

    # save_ply(
    #     merged_splat, 
    #     f_px=test_views[0]["focal_px"], 
    #     image_shape=(test_views[0]["height"], test_views[0]["width"]), 
    #     path=os.path.join(output_dir, "final.ply")
    # )

