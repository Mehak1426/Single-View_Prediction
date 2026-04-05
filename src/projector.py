"""
3D Point Cloud Projector.

Back-projects a metric depth map into 3D (X, Y, Z) coordinates using
the pinhole camera model, and exports the result as a .ply file.
"""

import os
import numpy as np
import open3d as o3d

from src.config import CONFIG


class PointCloudProjector:
    """
    Converts a 2D depth map into a colored 3D point cloud using
    the camera intrinsic matrix K and the pinhole projection model:

        Z = d_final(u, v)
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
    """

    def __init__(self, intrinsics: np.ndarray = None):
        """
        Args:
            intrinsics : (3, 3) camera matrix K.
                         If None, uses default NYU values from config.
        """
        if intrinsics is not None:
            self.K = intrinsics
        else:
            self.K = np.array([
                [CONFIG["fx"], 0.0,          CONFIG["cx"]],
                [0.0,          CONFIG["fy"], CONFIG["cy"]],
                [0.0,          0.0,          1.0         ],
            ])

    def deproject(self, depth: np.ndarray, rgb: np.ndarray = None):
        """
        Back-project every pixel into 3D space.

        Args:
            depth : (H, W) float32, metric depth in meters
            rgb   : (H, W, 3) uint8, optional color for each point

        Returns:
            points : (M, 3) float64 — 3D coordinates
            colors : (M, 3) float64 — normalized RGB in [0, 1], or None
        """
        h, w = depth.shape
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]

        # build pixel coordinate grids
        u = np.arange(w)
        v = np.arange(h)
        u, v = np.meshgrid(u, v)  # both are (H, W)

        # mask out invalid depths
        valid = (depth > 0) & np.isfinite(depth)

        Z = depth[valid]
        X = (u[valid] - cx) * Z / fx
        Y = (v[valid] - cy) * Z / fy

        points = np.stack([X, Y, Z], axis=-1)  # (M, 3)

        colors = None
        if rgb is not None:
            colors = rgb[valid].astype(np.float64) / 255.0

        return points, colors

    def to_pointcloud(self, depth: np.ndarray, rgb: np.ndarray = None):
        """
        Create an Open3D PointCloud object.

        Args:
            depth : (H, W) metric depth
            rgb   : (H, W, 3) uint8 color image

        Returns:
            pcd : open3d.geometry.PointCloud
        """
        points, colors = self.deproject(depth, rgb)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)

        return pcd

    def export_ply(
        self,
        depth: np.ndarray,
        rgb: np.ndarray = None,
        filepath: str = None,
    ) -> str:
        """
        Export the back-projected point cloud as a .ply file.

        Args:
            depth    : (H, W) metric depth
            rgb      : (H, W, 3) uint8, optional
            filepath : output path (defaults to outputs/pointcloud.ply)

        Returns:
            filepath : the path the .ply was saved to
        """
        if filepath is None:
            os.makedirs(CONFIG["output_dir"], exist_ok=True)
            filepath = os.path.join(CONFIG["output_dir"], CONFIG["ply_filename"])

        pcd = self.to_pointcloud(depth, rgb)
        o3d.io.write_point_cloud(filepath, pcd)
        print(f"[Projector] Saved {len(pcd.points)} points → {filepath}")

        return filepath
