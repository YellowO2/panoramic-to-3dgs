import torch
from depth_anything_3.api import DepthAnything3
from datatype import View

class DA3Model:
    def __init__(self, model_path="./models/models--depth-anything--DA3NESTED-GIANT-LARGE-1.1/snapshots/b2359bdf726fb44ef62acca04d629dcf158053e7", device="cuda"):
        self.device = device
        print(f"Loading Depth Anything 3 model from '{model_path}' on {device}...")
        self.model = DepthAnything3.from_pretrained(model_path).to(device=device)

    def process_views(self, views: list[View], export_dir=None):
        """
        Runs multi-view inference on the provided View objects 
        and updates them with the generated depth and extrinsics.
        """
        image_paths = [v.path for v in views]
        print(f"Running DA3 multi-view inference on {len(image_paths)} views...")
        
        if export_dir:
            prediction = self.model.inference(image_paths, export_dir=export_dir, export_format="glb")
        else:
            prediction = self.model.inference(image_paths)
            
        # Update the View objects with the results
        for i, v in enumerate(views):
            v.depth = prediction.depth[i]
            v.extrinsics = prediction.extrinsics[i]
            
        return prediction
