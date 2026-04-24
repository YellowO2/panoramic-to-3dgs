import os
import sys
import torch
import cv2
import numpy as np
from torchvision import transforms
 

class DA360DepthModel:
    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = device
        self.model = self._load_model(model_path)
        
    def _load_model(self, model_path):
        print(f"Loading DA360 model from '{model_path}' on {self.device}...")
        
        da360_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "DA360"))
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
        
        model.to(self.device)
        model_state_dict = model.state_dict()
        model.load_state_dict({k: v for k, v in model_dict.items() if k in model_state_dict}, strict=False)
        model.eval()
        return model

    def predict(self, image_path: str):
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not read image at {image_path}")
            
        original_h, original_w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to the model's training resolution
        img_resized = cv2.resize(img_rgb, (self.w, self.h), interpolation=cv2.INTER_LANCZOS4)
        
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        normalized_rgb = normalize(to_tensor(img_resized)).unsqueeze(0).to(self.device)

        # 3. Inference
        with torch.no_grad():
            model_to_use = self.model
            if torch.cuda.is_available() and self.device == "cuda":
                model_to_use = self.model.to(torch.bfloat16)           
                normalized_rgb = normalized_rgb.to(torch.bfloat16)  
            outputs = model_to_use(normalized_rgb)

        # 4. Convert Disp to Depth and scale
        pred_disp = outputs["pred_disp"].float().squeeze().cpu().numpy()
        pred_depth = 1.0 / (pred_disp + 1e-6)
        pred_depth = pred_depth / pred_depth.min() 
        
        # 5. Resize back exactly to the panoramas original resolution
        pred_depth = cv2.resize(pred_depth, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        
        return pred_depth

    def save_debug_ply(self, pred_depth, image_path, save_path):
        import open3d as o3d
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_h, original_w = img_rgb.shape[:2]
        
        print(f"Saving debug point cloud to {save_path}...")
        
        # Subsample for the debug point cloud to avoid massive file sizes (max width 1024)
        debug_w = min(1024, original_w)
        debug_h = int(original_h * (debug_w / original_w))
        
        debug_depth = cv2.resize(pred_depth, (debug_w, debug_h), interpolation=cv2.INTER_NEAREST)
        debug_rgb = cv2.resize(img_rgb, (debug_w, debug_h), interpolation=cv2.INTER_LINEAR)

        # Create full spherical mesh from the prediction
        h, w = debug_depth.shape
        Theta = np.pi - np.arange(h).reshape(h, 1) * np.pi / h - np.pi / (2 * h)
        Theta = np.repeat(Theta, w, axis=1)
        Phi = np.arange(w).reshape(1, w) * 2 * np.pi / w + np.pi / w - np.pi
        Phi = np.repeat(Phi, h, axis=0)

        # Spherical offset back to cartesian
        X = debug_depth * np.sin(Theta) * np.sin(Phi)
        Y = debug_depth * np.cos(Theta)
        Z = debug_depth * np.sin(Theta) * np.cos(Phi)
        
        median_depth = np.median(debug_depth)
        max_depth = median_depth * 2.0
        mask = debug_depth < max_depth
        
        X_m, Y_m, Z_m = X[mask], Y[mask], Z[mask]
        R_m, G_m, B_m = debug_rgb[:, :, 0][mask], debug_rgb[:, :, 1][mask], debug_rgb[:, :, 2][mask]
        
        points = np.stack([X_m, Y_m, Z_m], axis=1)
        colors = np.stack([R_m, G_m, B_m], axis=1) / 255.0
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(save_path, pcd)
