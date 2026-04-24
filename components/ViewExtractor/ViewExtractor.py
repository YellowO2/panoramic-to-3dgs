import os
import cv2 
import math
import Equirec2Perspec as E2P 
from datatype import View

def _save_view(equ, depth_equ, yaw, pitch, hfov, w, h, output_dir, filename) -> View:
    """Helper to slice a single view and return a View object."""
    img = equ.GetPerspective(hfov, yaw, pitch, h, w)
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, img)

    # Calculate focal length and vfov based on the provided hfov and width
    focal_px = (w / 2.0) / math.tan(math.radians(hfov) / 2.0)
    vfov = math.degrees(2.0 * math.atan((h / 2.0) / focal_px))
    
    view = View(
        yaw=yaw,
        pitch=pitch,
        path=output_path,
        width=int(w),
        height=int(h),
        focal_px=focal_px,
        hfov=hfov,
        vfov=vfov,
    )
    
    if depth_equ is not None:
        view.depth = depth_equ.GetPerspective(hfov, yaw, pitch, h, w)        
    return view

def extract_views(
    input_image,
    output_dir,
    overlap_degrees,
    slice_count=4,
    prefix="",
    panorama_depth=None,
) -> list[View]:
    """Extracts standard views for SHARP (side slices + top/bottom)."""
    equ = E2P.Equirectangular(input_image)
    depth_equ = E2P.Equirectangular(panorama_depth) if panorama_depth is not None else None

    pano_h, pano_w = equ._img.shape[:2]
    slice_w = max(64, pano_w // slice_count)
    slice_h = pano_h * 0.8

    span_degrees = 360.0 / slice_count
    hfov = min(170.0, span_degrees + float(overlap_degrees))
    
    print(f"Extracting main views: {slice_w}x{slice_h} | Slices: {slice_count} | HFOV: {hfov:.2f}")

    views_data: list[View] = []
    
    # 1. Side views
    yaw_values = [(span_degrees * i) for i in range(slice_count)]
    for yaw in yaw_values:
        filename = f"{prefix}view_{int(round(yaw))}_0.jpg"
        views_data.append(_save_view(equ, depth_equ, yaw, 0, hfov, slice_w, slice_h, output_dir, filename))

    # 2. Top/Bottom views (usually square for zenith/nadir)
    tb_hfov = 60.0
    tb_size = slice_w
    
    views_data.append(_save_view(equ, depth_equ, 0, 90, tb_hfov, tb_size, tb_size, output_dir, f"{prefix}view_0_90.jpg"))
    views_data.append(_save_view(equ, depth_equ, 0, -90, tb_hfov, tb_size, tb_size, output_dir, f"{prefix}view_0_-90.jpg"))
    
    return views_data

def extract_views_for_da360(
    input_image,
    output_dir,
    overlap_degrees,
    slice_count,
    prefix="",
    panorama_depth=None,
) -> list[View]:
    """Extracts high-overlap views for DA360 alignment."""
    equ = E2P.Equirectangular(input_image)
    depth_equ = E2P.Equirectangular(panorama_depth) if panorama_depth is not None else None

    pano_h, pano_w = equ._img.shape[:2]
    slice_w = max(64, pano_w // slice_count)
    slice_h = slice_w / 16 * 9 # 16:9 aspect ratio

    span_degrees = 360.0 / slice_count
    hfov = min(170.0, span_degrees + float(overlap_degrees))
    
    print(f"Extracting DA360 views: {slice_w}x{slice_h} | Slices: {slice_count} | HFOV: {hfov:.2f}")

    views_data = []
    yaw_values = [(span_degrees * i) for i in range(slice_count)]
    for yaw in yaw_values:
        filename = f"{prefix}_{int(round(yaw))}_0.jpg"
        views_data.append(_save_view(equ, depth_equ, yaw, 0, hfov, slice_w, slice_h, output_dir, filename))
            
    return views_data
