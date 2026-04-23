import os
import cv2
import torch
from sharp.cli.predict import predict_image
from sharp.models import PredictorParams, create_predictor
from sharp.utils.gaussians import save_ply
from datatype import View

# check for cpu or cuda to use
def _resolve_device(device: str = None):
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


# load sharp model
def load_sharp_predictor(model_path: str, device: str = None):
    device_obj = _resolve_device(device)
    state_dict = torch.load(model_path, weights_only=True)
    predictor = create_predictor(PredictorParams())
    predictor.load_state_dict(state_dict)
    predictor.eval().to(device_obj)
    return predictor, device_obj


# a wrapper around SHARP functions that turns extracted views into 3DGS.
def extracted_views_to_3dgs(
    extracted_views: list[View],
    *,
    model_path: str,
    output_dir: str,
    device: str = None,
): 
    predictor, device_obj = load_sharp_predictor(model_path, device=device)

    ply_dir = os.path.join(output_dir, "ply")
    os.makedirs(ply_dir, exist_ok=True)

    gaussian_list = []

    for v in extracted_views:
        img_bgr = cv2.imread(v.path, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        current_focal_px = float(v.focal_px)
        gaussians = predict_image(predictor, img_rgb, current_focal_px, device_obj)

        ply_path = os.path.join(
            ply_dir,
            f"view_{int(round(v.yaw))}_{int(round(v.pitch))}.ply",
        )
        save_ply(gaussians, current_focal_px, (v.height, v.width), ply_path)
        gaussian_list.append(gaussians)
    
    return gaussian_list