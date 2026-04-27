import os
import numpy as np
import cv2
import open3d as o3d

class Saver:
    @staticmethod
    def colorize_depth(depth):
        """Turns raw depth (meters/units) into a colored image (JPG/PNG compatible)."""
        # Normalize to 0-255
        depth_min = depth.min()
        depth_max = depth.max()
        if depth_max - depth_min > 0:
            normalized = (depth - depth_min) / (depth_max - depth_min)
        else:
            normalized = depth * 0
        
        # Apply a standard colormap (Magma or Jet)
        colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
        return colored

    @staticmethod
    def save_depth_image(depth, path):
        """Saves depth as a colored JPG/PNG."""
        colored = Saver.colorize_depth(depth)
        cv2.imwrite(path, colored)
        print(f"Saved depth image to: {path}")

    @staticmethod
    def save_point_cloud(points: np.ndarray, path: str, colors: np.ndarray = None):
        """
        Saves a raw XYZ point cloud (N, 3) to a PLY file.
        Args:
            points: (N, 3) array of XYZ points.
            path: Output path.
            colors: (N, 3) normalized RGB colors OR (H, W, 3) image.
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        if colors is not None:
            if colors.ndim == 2 and colors.shape[0] == points.shape[0]:
                # Already a list of colors (normalized 0-1)
                pcd.colors = o3d.utility.Vector3dVector(colors)
            elif colors.ndim == 3:
                # It's an image, need to reshape it (legacy support)
                # Assuming colors is BGR image from cv2
                img_rgb = cv2.cvtColor(colors, cv2.COLOR_BGR2RGB)
                flat_colors = img_rgb.reshape(-1, 3) / 255.0
                if len(flat_colors) == len(points):
                    pcd.colors = o3d.utility.Vector3dVector(flat_colors)
                else:
                    print(f"Warning: Image size doesn't match point count. Skipping colors.")
        
        o3d.io.write_point_cloud(path, pcd)
        print(f"Saved point cloud to: {path}")
