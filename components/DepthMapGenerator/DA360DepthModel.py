import os
import sys
from typing import Optional, Tuple
import torch
import cv2
import numpy as np
from torchvision import transforms

class DA360DepthModel:
    def __init__(self, model_path: str, device: str = "cuda") -> None:
        self.device = device
        self.model = self._load_model(model_path)
        
    def _load_model(self, model_path: str) -> torch.nn.Module:
        print(f"Loading DA360 model from '{model_path}' on {self.device}...")

        # Point to the local modularized implementation
        da360_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "da360"))
        if da360_root not in sys.path:
            sys.path.insert(0, da360_root)

        import networks 

        model_dict = torch.load(model_path, map_location="cpu")
        net_type = model_dict.get('net', 'DA360')
        dinov2_encoder = model_dict.get('dinov2_encoder', 'vits')
        self.h = model_dict.get('height', 518)
        self.w = model_dict.get('width', 1036)
        
        Net = getattr(networks, net_type)
        model = Net(self.h, self.w, dinov2_encoder=dinov2_encoder)
        
        model_state_dict = model.state_dict()
        model.load_state_dict({k: v for k, v in model_dict.items() if k in model_state_dict}, strict=False)
        
        model.to(self.device)
        model.eval()
        return model

    def predict(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not read image at {image_path}")
            
        original_h, original_w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img_resized = cv2.resize(img_rgb, (self.w, self.h), interpolation=cv2.INTER_LANCZOS4)
        
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        normalized_rgb = normalize(to_tensor(img_resized)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(normalized_rgb)

        # Matched exactly to author's math
        pred_disp = outputs["pred_disp"].detach().cpu().squeeze().numpy()
        pred_depth = 1.0 / pred_disp
        pred_depth = pred_depth / pred_depth.min() 
        
        pred_depth = cv2.resize(pred_depth, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        
        return pred_depth, img_rgb

    @staticmethod
    def to_world_pts(
        depth: np.ndarray,
        image: Optional[np.ndarray] = None,
        v_fov_deg: float = None,
        max_depth_mult: float = 4.0,
        max_pts: int = 6_000_000,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Backproject an equirectangular depth map to a 3D point cloud."""
        h, w = depth.shape
        stride = max(1, int((h * w / max_pts) ** 0.5))
        depth = depth[::stride, ::stride]
        if image is not None:
            image = image[::stride, ::stride]
        h, w = depth.shape
        if v_fov_deg is None:
            theta_start, theta_end = 0.0, np.pi
        else:
            v_fov_rad = np.radians(v_fov_deg)
            theta_start = np.pi / 2.0 - v_fov_rad / 2.0
            theta_end = np.pi / 2.0 + v_fov_rad / 2.0

        theta = np.linspace(theta_start, theta_end, h)
        theta_grid = np.repeat(theta.reshape(h, 1), w, axis=1)
        phi = np.linspace(-np.pi, np.pi, w)
        phi_grid = np.repeat(phi.reshape(1, w), h, axis=0)

        x = depth * np.sin(theta_grid) * np.sin(phi_grid)
        y = -depth * np.cos(theta_grid)  # negate: theta=0 (top) → y=-depth (up = negative in Y-down)
        z = depth * np.sin(theta_grid) * np.cos(phi_grid)
        points = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)

        colors = None
        if image is not None:
            colors = image.reshape(-1, 3) / 255.0

        d_flat = depth.flatten()
        valid = d_flat > 1e-3
        if np.any(valid):
            valid &= d_flat < (np.median(d_flat[valid]) * max_depth_mult)

        return points[valid], (colors[valid] if colors is not None else None)