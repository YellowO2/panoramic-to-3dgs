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
    def save_point_cloud(depth, rgb_path, path):
        """Saves depth + original image as a 3D PLY file."""
        img = cv2.imread(rgb_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = depth.shape

        # Create 3D points
        Theta = np.pi - np.arange(h).reshape(h, 1) * np.pi / h - np.pi / (2 * h)
        Theta = np.repeat(Theta, w, axis=1)
        Phi = np.arange(w).reshape(1, w) * 2 * np.pi / w + np.pi / w - np.pi
        Phi = np.repeat(Phi, h, axis=0)

        X = depth * np.sin(Theta) * np.sin(Phi)
        Y = depth * np.cos(Theta)
        Z = depth * np.sin(Theta) * np.cos(Phi)
        
        # Simple mask to remove very far points (sky)
        mask = depth < (np.median(depth) * 5.0)

        XYZ = np.stack([X[mask], Y[mask], Z[mask]], axis=1)
        RGB = np.stack([img_rgb[:, :, 0][mask], img_rgb[:, :, 1][mask], img_rgb[:, :, 2][mask]], axis=1) / 255.0

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(XYZ)
        pcd.colors = o3d.utility.Vector3dVector(RGB)
        o3d.io.write_point_cloud(path, pcd)
        print(f"Saved point cloud to: {path}")
