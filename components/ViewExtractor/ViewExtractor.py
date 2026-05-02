import os
import cv2
import math
from components.ViewExtractor import Equirec2Perspec as E2P
from datatype import View


def _extract_slice(
    equ, yaw, pitch, hfov, w, h, output_path, pano_id, depth_equ=None
) -> View:
    """Extract one perspective slice, save it, and return a View."""
    img = equ.GetPerspective(hfov, yaw, pitch, h, w)
    cv2.imwrite(output_path, img)

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
        pano_id=pano_id,
    )
    if depth_equ is not None:
        view.depth = depth_equ.GetPerspective(hfov, yaw, pitch, h, w)
    return view


def extract_views(
    input_image,
    output_dir,
    overlap_degrees=0,
    slice_count=4,
    prefix="",
    panorama_depth=None,
    pano_id=0,
) -> list[View]:
    """Extracts standard views for SHARP: horizon slices + top/bottom poles."""
    equ = E2P.Equirectangular(input_image)
    depth_equ = (
        E2P.Equirectangular(panorama_depth) if panorama_depth is not None else None
    )
    pano_h, pano_w = equ._img.shape[:2]

    slice_w = max(64, pano_w // slice_count)
    slice_h = pano_h * 0.5
    horizon_hfov = min(170.0, (360.0 / slice_count) + float(overlap_degrees))

    views = []
    for i in range(slice_count):
        yaw = i * (360.0 / slice_count)
        filename = f"{prefix}sharp_{int(round(yaw))}_0.jpg"
        views.append(
            _extract_slice(
                equ,
                yaw,
                0,
                horizon_hfov,
                slice_w,
                slice_h,
                os.path.join(output_dir, filename),
                pano_id,
                depth_equ,
            )
        )

    pole_size = slice_w
    # generate bottom (ground) view only; sky omitted
    # views.append(_extract_slice(equ, 0,  90, 60.0, pole_size, pole_size,
    #                             os.path.join(output_dir, f"{prefix}sharp_0_90.jpg"), pano_id, depth_equ))
    views.append(
        _extract_slice(
            equ,
            0,
            -90,
            90.0,
            pole_size,
            pole_size,
            os.path.join(output_dir, f"{prefix}sharp_0_-90.jpg"),
            pano_id,
            depth_equ,
        )
    )
    return views


HFOV = 90.0  # Fixed FOV for DA3 slices


def extract_views_for_da3(
    input_image, output_dir, step_degrees=20, prefix="", pano_id=0
) -> list[View]:
    """Extracts 16:9 horizon-only views optimized for Depth Anything 3."""
    equ = E2P.Equirectangular(input_image)
    pano_w = equ._img.shape[1]

    slice_w = max(64, pano_w // 4)
    slice_h = int(slice_w * 9 / 16)

    views = []
    yaw = 0.0
    while yaw < 360.0:
        filename = f"{prefix}da3_{int(round(yaw))}_0.jpg"
        views.append(
            _extract_slice(
                equ,
                yaw,
                0,
                HFOV,
                slice_w,
                slice_h,
                os.path.join(output_dir, filename),
                pano_id,
            )
        )
        yaw += step_degrees
    return views
