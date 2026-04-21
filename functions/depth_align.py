import torch
import numpy as np
import cv2
from sharp.utils.gaussians import Gaussians3D
from depth_anything_3.api import DepthAnything3

def get_da360_panorama_depth(image_path: str, model_path: str = "models/DA360_large.pth", device: str = "cuda"):
    """
    Loads DA360 and generates a full equirectangular depth map from the panorama.
    """
    print(f"Loading DA360 model from '{model_path}' on {device}...")
    
    import sys
    import os
    # Temporarily add DA360 to system path so we can import its modules natively
    da360_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "third_party", "DA360"))
    if da360_root not in sys.path:
        sys.path.insert(0, da360_root)
        
    import networks 

    # 1. Load the pre-trained dictionary and instantiate the model
    model_dict = torch.load(model_path, map_location="cpu")
    net_type = model_dict.get('net', 'DA360')
    dinov2_encoder = model_dict.get('dinov2_encoder', 'vits')
    h = model_dict.get('height', 518)
    w = model_dict.get('width', 1036)
    
    Net = getattr(networks, net_type)
    model = Net(h, w, dinov2_encoder=dinov2_encoder)
    
    model.to(device)
    model_state_dict = model.state_dict()
    model.load_state_dict({k: v for k, v in model_dict.items() if k in model_state_dict}, strict=False)
    model.eval()

    # 2. Load and format image
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    original_h, original_w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize to the model's training resolution
    img_resized = cv2.resize(img_rgb, (w, h), interpolation=cv2.INTER_LANCZOS4)
    
    from torchvision import transforms
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    normalized_rgb = normalize(to_tensor(img_resized)).unsqueeze(0).to(device)

    # 3. Inference
    with torch.no_grad():
        if torch.cuda.is_available() and device == "cuda":
            model = model.to(torch.bfloat16)           
            normalized_rgb = normalized_rgb.to(torch.bfloat16)  
        outputs = model(normalized_rgb)

    # 4. Convert Disp to Depth and scale
    pred_disp = outputs["pred_disp"].float().squeeze().cpu().numpy()
    pred_depth = 1.0 / (pred_disp + 1e-6)
    pred_depth = pred_depth / pred_depth.min() 
    
    # 5. Resize back exactly to the panoramas original resolution
    pred_depth = cv2.resize(pred_depth, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
    
    del model, outputs, normalized_rgb
    torch.cuda.empty_cache()
    
    return pred_depth


def get_da3_predictions(image_paths, export_dir=None, model_type="./models/models--depth-anything--DA3NESTED-GIANT-LARGE-1.1/snapshots/b2359bdf726fb44ef62acca04d629dcf158053e7", device="cuda"):
    print(f"Loading Depth Anything 3 model '{model_type}' on {device}...")
    model = DepthAnything3.from_pretrained(model_type).to(device=device)
    
    print(f"Running DA3 inference on {len(image_paths)} views...")
    if export_dir:
        print(f"Exporting GLB to {export_dir}...")
        prediction = model.inference(image_paths, export_dir=export_dir, export_format="glb")
    else:
        prediction = model.inference(image_paths)
    return prediction


def scale_splat_to_depthmap(gaussians: Gaussians3D, ref_depth_map: np.ndarray, focal_px: float, width: int, height: int) -> Gaussians3D:
    # === 1. project 3dgs to depthmap ===
    positions = gaussians.mean_vectors[0] # Shape: (N, 3)
    z = positions[:, 2]
    valid = z > 0.1
    # convert to pixel coordinates
    px = (positions[:, 0] / z) * focal_px + (width / 2.0)
    py = (positions[:, 1] / z) * focal_px + (height / 2.0)
    valid = valid & (px >= 0) & (px < width - 1) & (py >= 0) & (py < height - 1)
    
    valid_z = z[valid].detach().cpu().numpy()
    valid_px = px[valid].long().detach().cpu().numpy()
    valid_py = py[valid].long().detach().cpu().numpy()
    
    # === 2. Obtain depthmap from DA3 ===
    ref_depth_map_resized = cv2.resize(ref_depth_map, (width, height), interpolation=cv2.INTER_LINEAR)
    da3_z = ref_depth_map_resized[valid_py, valid_px]
    # === 3. Compute scale ratio ===
    ratios = da3_z / valid_z
    median_scale = float(np.median(ratios)) # we use median to reduce influence from outliers
    print(f"Calculated scale ratio: {median_scale:.4f}")
    
    # === 4. Apply the scale ===
    # scale both the position & size of the blobs
    scaled_mean_vectors = gaussians.mean_vectors * median_scale
    scaled_singular_values = gaussians.singular_values * median_scale
    return Gaussians3D(
        mean_vectors=scaled_mean_vectors,
        singular_values=scaled_singular_values,
        quaternions=gaussians.quaternions,
        colors=gaussians.colors,
        opacities=gaussians.opacities,
    )

def align_gaussians_to_reference(
    gaussians: Gaussians3D,
    reference_depth_view: np.ndarray,
    focal_x_px: float,
    focal_y_px: float,
    image_width: int,
    image_height: int,
    grid_resolution: int = 8,
    detail_weight: float = 0.0,
) -> tuple[Gaussians3D, float, int]:
  
    # Scale the positions and scales uniformly by the chosen factor.
    aligned_gaussians = scale_splat_to_depthmap(gaussians, reference_depth_view, focal_x_px, image_width, image_height)
    return aligned_gaussians

