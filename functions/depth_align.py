import torch
import numpy as np
import cv2
from sharp.utils.gaussians import Gaussians3D
from depth_anything_3.api import DepthAnything3

def get_da360_panorama_depth(image_path: str, model_path: str = "models/DA360_large.pth", device: str = "cuda", save_debug_ply: str = None):
    """
    Loads DA360 and generates a full equirectangular depth map from the panorama.
    If save_debug_ply is provided (e.g. "debug_da360.ply"), a point cloud will be saved.
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
    
    if save_debug_ply is not None:
        print(f"Saving debug point cloud to {save_debug_ply}...")
        
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
        
        # Mask out background/sky (usually extreme depths > some threshold)
        mask = debug_depth < 200.0
        X_m, Y_m, Z_m = X[mask], Y[mask], Z[mask]
        R_m, G_m, B_m = debug_rgb[:, :, 0][mask], debug_rgb[:, :, 1][mask], debug_rgb[:, :, 2][mask]
        
        # Format the coordinates as standard Open3D structures
        import open3d as o3d
        points = np.stack([X_m, Y_m, Z_m], axis=1)
        colors = np.stack([R_m, G_m, B_m], axis=1) / 255.0  # Open3D colors must be 0-1 floats
        
        # Create and save a strict standard PLY format that web viewers can read safely
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(save_debug_ply, pcd)

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
    """Align Gaussian depths to DA360 using a smooth low-frequency scale field."""
    
    grid_cells_x = max(1, int(grid_resolution))
    grid_cells_y = max(1, int(round(grid_cells_x * (image_height / max(1, image_width)))))

    mean_vectors = gaussians.mean_vectors  # (1, N, 3)
    mv_np = mean_vectors[0].detach().cpu().numpy().astype(np.float32)
    depth_z = mv_np[:, 2]
    
    # Use radial depth (distance from camera center), not planar Z-depth.
    # Panoramic models generate spherical radial distance for every pixel.
    radial = np.linalg.norm(mv_np, axis=1)
    
    valid = depth_z > 1e-6
    pixel_x = (mv_np[:, 0] / np.clip(depth_z, 1e-6, None)) * focal_x_px + (image_width / 2.0) - 0.5
    pixel_y = (mv_np[:, 1] / np.clip(depth_z, 1e-6, None)) * focal_y_px + (image_height / 2.0) - 0.5
    valid &= (pixel_x >= 0) & (pixel_x <= image_width - 1)
    valid &= (pixel_y >= 0) & (pixel_y <= image_height - 1)

    per_point_scale = np.ones(mv_np.shape[0], dtype=np.float32)
    median_scale = 1.0
    count = 0

    if int(valid.sum()) >= 64:
        # Sample the depth map at pixel locations
        px_int = np.clip(np.round(pixel_x[valid]).astype(np.int32), 0, image_width - 1)
        py_int = np.clip(np.round(pixel_y[valid]).astype(np.int32), 0, image_height - 1)
        
        # Note: DA360 generates depth directly, not disparity for us, because of the 1/disp inversion earlier
        ref_depth_sampled = reference_depth_view[py_int, px_int]
        
        ok = np.isfinite(ref_depth_sampled) & (ref_depth_sampled > 1e-6) & (radial[valid] > 1e-6)
        count = int(ok.sum())
        if count >= 64:
            ref_depth_ok = ref_depth_sampled[ok].astype(np.float32)
            sharp_r_ok = radial[valid][ok]
            
            # The critical fix! We must compare DA360 radial depth to SHARP radial depth.
            raw_scale = ref_depth_ok / sharp_r_ok

            # Global robust median for fallback and logging.
            lo, hi = np.quantile(raw_scale, [0.05, 0.95])
            trimmed = raw_scale[(raw_scale >= lo) & (raw_scale <= hi)]
            median_scale = (
                float(np.median(trimmed)) if trimmed.size > 0
                else float(np.median(raw_scale))
            )

            # --- Build coarse scale grid ---
            px_ok = pixel_x[valid][ok]
            py_ok = pixel_y[valid][ok]

            cell_width = image_width / grid_cells_x
            cell_height = image_height / grid_cells_y
            grid = np.full((grid_cells_y, grid_cells_x), median_scale, dtype=np.float32)
            for gy in range(grid_cells_y):
                for gx in range(grid_cells_x):
                    in_cell = (
                        (px_ok >= gx * cell_width) & (px_ok < (gx + 1) * cell_width)
                        & (py_ok >= gy * cell_height) & (py_ok < (gy + 1) * cell_height)
                    )
                    if int(in_cell.sum()) >= 8:
                        cell_scales = raw_scale[in_cell]
                        cl, ch = np.quantile(cell_scales, [0.1, 0.9])
                        cell_trimmed = cell_scales[(cell_scales >= cl) & (cell_scales <= ch)]
                        if cell_trimmed.size > 0:
                            grid[gy, gx] = float(np.median(cell_trimmed))

            # Clamp extreme cells relative to global median.
            grid = np.clip(grid, median_scale * 0.1, median_scale * 10.0)

            # --- Interpolate coarse grid to every Gaussian ---
            all_px = np.clip(pixel_x, 0, image_width - 1)
            all_py = np.clip(pixel_y, 0, image_height - 1)
            gx_cont = all_px / cell_width - 0.5
            gy_cont = all_py / cell_height - 0.5

            gx0 = np.clip(np.floor(gx_cont).astype(np.int32), 0, grid_cells_x - 1)
            gy0 = np.clip(np.floor(gy_cont).astype(np.int32), 0, grid_cells_y - 1)
            gx1 = np.clip(gx0 + 1, 0, grid_cells_x - 1)
            gy1 = np.clip(gy0 + 1, 0, grid_cells_y - 1)
            wx = np.clip(gx_cont - gx0, 0, 1).astype(np.float32)
            wy = np.clip(gy_cont - gy0, 0, 1).astype(np.float32)

            s00 = grid[gy0, gx0]
            s01 = grid[gy0, gx1]
            s10 = grid[gy1, gx0]
            s11 = grid[gy1, gx1]
            smooth_scale = (
                s00 * (1 - wx) * (1 - wy)
                + s01 * wx * (1 - wy)
                + s10 * (1 - wx) * wy
                + s11 * wx * wy
            )

            # Blend smooth grid scale with per-point raw scale.
            dw = float(np.clip(detail_weight, 0.0, 1.0))
            if dw > 0.0:
                per_point_raw = np.full(mv_np.shape[0], median_scale, dtype=np.float32)
                valid_indices = np.where(valid)[0]
                ok_within_valid = np.where(ok)[0]
                per_point_raw[valid_indices[ok_within_valid]] = raw_scale
                per_point_scale = smooth_scale * (1.0 - dw) + per_point_raw * dw
            else:
                per_point_scale = smooth_scale

    # === Apply the scale array to the splats ===
    # Convert the (N,) scale array to tensor matching the splat device
    device = gaussians.mean_vectors.device
    scale_tensor = torch.tensor(per_point_scale, dtype=torch.float32, device=device).unsqueeze(1) # (N, 1)
    
    scaled_mean_vectors = gaussians.mean_vectors * scale_tensor
    scaled_singular_values = gaussians.singular_values * scale_tensor
    
    print(f"Applied grid-based median scale alignment. Median global fallback scale: {median_scale:.4f}")
    
    aligned_gaussians = Gaussians3D(
        mean_vectors=scaled_mean_vectors,
        singular_values=scaled_singular_values,
        quaternions=gaussians.quaternions,
        colors=gaussians.colors,
        opacities=gaussians.opacities,
    )
    return aligned_gaussians

# implement over here.