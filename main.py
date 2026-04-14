import os
import cv2 
import math
from functions.sharp_infer import extracted_views_to_3dgs
from functions import Equirec2Perspec as E2P 

# cuts the panoramic into correct inputs for MLsharp
def extract_views(
    input_image,
    output_dir,
    slice_count=4,
    overlap_degrees=10.0,
):
    equ = E2P.Equirectangular(input_image)  # load panorama image

    pano_h, pano_w = equ._img.shape[:2]
    slice_w = max(64, pano_w // slice_count)
    slice_h = pano_h

    span_degrees = 360.0 / slice_count
    hfov = span_degrees + float(overlap_degrees)
    hfov = min(170.0, hfov)

    focal_px = (slice_w / 2.0) / math.tan(math.radians(hfov) / 2.0)
    vfov = math.degrees(2.0 * math.atan((slice_h / 2.0) / focal_px))
    print(
        f"Panorama: {pano_w}x{pano_h} | slices: {slice_count} | slice: {slice_w}x{slice_h} | "
        f"HFOV: {hfov:.2f} | VFOV: {vfov:.2f} | focal_px: {focal_px:.2f}"
    )

    yaw_values = [(span_degrees * i) for i in range(slice_count)] # step through the size of a slice without overlaps.

    views = []

    # generate the side slices first.
    for yaw in yaw_values:
        # 90 degree fov means top 45 + bottom 45 here
        # pitch refers to vertical axis, where 0 is center of image. The max top is 90 and bottom -90
        # yaw refers to horizontal axis. The max right is 180 and left -180.
        pitch = 0
        img = equ.GetPerspective(hfov, yaw, pitch, slice_h, slice_w)
        output_path = os.path.join(output_dir, f"view_{int(round(yaw))}_{int(round(pitch))}.jpg")
        cv2.imwrite(output_path, img)

        views.append(
            {
                "yaw": yaw,
                "pitch": pitch,
                "path": output_path,
                "width": slice_w,
                "height": slice_h,
                "focal_px": focal_px,
            }
        )
    # --- top view slice. It is a square. ---
    top_bottom_hfov = 150.0 
    top_bottom_focal_px = (slice_w / 2.0) / math.tan(math.radians(top_bottom_hfov) / 2.0)
    img_top = equ.GetPerspective(top_bottom_hfov, 0, 90, slice_w, slice_w)
    path_top = os.path.join(output_dir, "view_0_90.jpg")
    cv2.imwrite(path_top, img_top)
    views.append({
        "yaw": 0, "pitch": 90, 
        "path": path_top, "width": slice_w, "height": slice_w, "focal_px": top_bottom_focal_px
    })

    # --- bottom view slice ---
    img_bottom = equ.GetPerspective(top_bottom_hfov, 0, -90, slice_w, slice_w)
    path_bottom = os.path.join(output_dir, "view_0_-90.jpg")
    cv2.imwrite(path_bottom, img_bottom)
    views.append({
        "yaw": 0, "pitch": -90, 
        "path": path_bottom, "width": slice_w, "height": slice_w, "focal_px": top_bottom_focal_px
    })
    return views



if __name__ == '__main__':
    output_dir = 'output_views'
    os.makedirs(output_dir, exist_ok=True) 
    views = extract_views(
        'panorama_test.jpg',
        output_dir,
        slice_count=4,
        overlap_degrees=10.0,
    )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = 'output_3dgs'
    model_path = os.path.join(script_dir, "models", "sharp_2572gikvuh.pt")
    
    extracted_views_to_3dgs(
        views,
        model_path=model_path,
        output_dir=output_dir,
        # can specify device param if what to force CPU when u have GPU.
    )