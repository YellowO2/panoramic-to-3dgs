import os
import cv2 
import math
import numpy as np
from components.ViewExtractor import Equirec2Perspec as E2P
from datatype import View

def _save_view(equ, depth_equ, yaw, pitch, hfov, w, h, output_dir, filename, pano_id=0) -> View:
    """Helper to slice a single view and return a View object."""
    img = equ.GetPerspective(hfov, yaw, pitch, h, w)
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, img)

    # Calculate focal length and vfov
    focal_px = (w / 2.0) / math.tan(math.radians(hfov) / 2.0)
    vfov = math.degrees(2.0 * math.atan((h / 2.0) / focal_px))
    
    view = View(
        yaw=yaw, pitch=pitch, path=output_path,
        width=int(w), height=int(h), focal_px=focal_px,
        hfov=hfov, vfov=vfov, pano_id=pano_id
    )
    
    if depth_equ is not None:
        view.depth = depth_equ.GetPerspective(hfov, yaw, pitch, h, w)        
    return view

def extract_views(
    input_image, output_dir, overlap_degrees=0, slice_count=4, prefix="", panorama_depth=None, pano_id=0
) -> list[View]:
    """Extracts standard 2:1 or square views for SHARP (Horizon + Top/Bottom)."""
    equ = E2P.Equirectangular(input_image)
    depth_equ = E2P.Equirectangular(panorama_depth) if panorama_depth is not None else None
    pano_h, pano_w = equ._img.shape[:2]
    
    # SHARP standard: wide side views
    slice_w = max(64, pano_w // slice_count)
    slice_h = pano_h * 0.5 
    hfov = min(170.0, (360.0 / slice_count) + float(overlap_degrees))
    
    views = []
    # 1. Horizon
    for i in range(slice_count):
        yaw = i * (360.0 / slice_count)
        views.append(_save_view(equ, depth_equ, yaw, 0, hfov, slice_w, slice_h, output_dir, f"{prefix}sharp_{int(round(yaw))}_0.jpg", pano_id))
    
    # 2. Top/Bottom
    tb_hfov, tb_size = 60.0, slice_w
    views.append(_save_view(equ, depth_equ, 0, 90, tb_hfov, tb_size, tb_size, output_dir, f"{prefix}sharp_0_90.jpg", pano_id))
    views.append(_save_view(equ, depth_equ, 0, -90, tb_hfov, tb_size, tb_size, output_dir, f"{prefix}sharp_0_-90.jpg", pano_id))
    return views

def extract_views_for_da3(
    input_image, output_dir, slice_count=8, prefix="", pano_id=0
) -> list[View]:
    """Extracts 16:9 horizon-only views optimized for Depth Anything 3."""
    equ = E2P.Equirectangular(input_image)
    pano_h, pano_w = equ._img.shape[:2]
    
    # DA3 optimized: 16:9 aspect ratio
    slice_w = max(64, pano_w // 4) # Base width on 4-slice coverage for resolution
    slice_h = int(slice_w * 9 / 16)
    
    # Use higher slice count (8) for better multi-view matching consistency in DA3
    hfov = (360.0 / slice_count) * 1.5 # 50% overlap for robust matching
    
    views = []
    for i in range(slice_count):
        yaw = i * (360.0 / slice_count)
        views.append(_save_view(equ, None, yaw, 0, hfov, slice_w, slice_h, output_dir, f"{prefix}da3_{int(round(yaw))}_0.jpg", pano_id))
    return views
