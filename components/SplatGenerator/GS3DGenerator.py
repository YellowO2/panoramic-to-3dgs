import os
import cv2
import torch
from sharp.cli.predict import predict_image
from sharp.models import PredictorParams, create_predictor
from sharp.utils.gaussians import save_ply
from datatype import View

class GS3DGenerator:
    def __init__(self, model_path: str, device: str = None):
        self.device = self._resolve_device(device)
        self.predictor = self._load_sharp_predictor(model_path)

    def _resolve_device(self, device: str = None):
        if device:
            return torch.device(device)
        return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    def _load_sharp_predictor(self, model_path: str):
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        predictor = create_predictor(PredictorParams())
        predictor.load_state_dict(state_dict)
        predictor.eval().to(self.device)
        return predictor

    def generate_from_view(self, view: View, save_ply_path: str = None):
        img_bgr = cv2.imread(view.path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError(f"Could not read image at {view.path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        current_focal_px = float(view.focal_px)
        gaussians = predict_image(self.predictor, img_rgb, current_focal_px, self.device)
        
        if save_ply_path:
            os.makedirs(os.path.dirname(save_ply_path), exist_ok=True)
            save_ply(gaussians, current_focal_px, (view.height, view.width), save_ply_path)
            
        view.splat = gaussians
        return gaussians

    def generate_from_views(self, views: list[View], output_dir: str = None):
        gaussian_list = []
        for v in views:
            save_path = None
            if output_dir:
                ply_dir = os.path.join(output_dir, "ply")
                save_path = os.path.join(ply_dir, f"view_{int(round(v.yaw))}_{int(round(v.pitch))}.ply")
            
            gaussians = self.generate_from_view(v, save_ply_path=save_path)
            gaussian_list.append(gaussians)
        return gaussian_list
