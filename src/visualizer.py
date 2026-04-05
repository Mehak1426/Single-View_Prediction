"""
Visualization utilities.

Provides helper functions for:
  - Side-by-side depth map comparison plots
  - Open3D interactive point cloud viewer
  - Saving result figures to disk
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from src.config import CONFIG


def plot_depth_comparison(
    rgb: np.ndarray,
    depth_pred: np.ndarray,
    depth_aligned: np.ndarray,
    depth_gt: np.ndarray,
    save_path: str = None,
    title: str = "Depth Comparison",
):
    """
    Plot a 1×4 grid:  RGB | Predicted | Aligned | Ground Truth.

    Args:
        rgb           : (H, W, 3) uint8
        depth_pred    : (H, W) raw monocular prediction
        depth_aligned : (H, W) after scale-shift fusion
        depth_gt      : (H, W) ground-truth metric depth
        save_path     : if given, saves the figure to this path
        title         : figure super-title
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(title, fontsize=14)

    axes[0].imshow(rgb)
    axes[0].set_title("Input RGB")
    axes[0].axis("off")

    vmin = np.percentile(depth_gt[depth_gt > 0], 2)
    vmax = np.percentile(depth_gt[depth_gt > 0], 98)

    axes[1].imshow(depth_pred, cmap="inferno")
    axes[1].set_title("Predicted (Relative)")
    axes[1].axis("off")

    axes[2].imshow(depth_aligned, cmap="inferno", vmin=vmin, vmax=vmax)
    axes[2].set_title("Aligned (Metric)")
    axes[2].axis("off")

    axes[3].imshow(depth_gt, cmap="inferno", vmin=vmin, vmax=vmax)
    axes[3].set_title("Ground Truth")
    axes[3].axis("off")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Visualizer] Saved figure → {save_path}")

    plt.show()


def plot_error_map(
    depth_aligned: np.ndarray,
    depth_gt: np.ndarray,
    save_path: str = None,
):
    """
    Plot the absolute error map between aligned and GT depth.
    """
    valid = (depth_gt > 0) & np.isfinite(depth_gt)
    error = np.zeros_like(depth_gt)
    error[valid] = np.abs(depth_aligned[valid] - depth_gt[valid])

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    im = ax.imshow(error, cmap="hot")
    ax.set_title("Absolute Error Map (meters)")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Visualizer] Saved error map → {save_path}")

    plt.show()


def plot_sparsity_curve(
    n_values: list,
    rmse_values: list,
    save_path: str = None,
):
    """
    Plot the sparsity sensitivity curve: RMSE vs. number of anchors.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(n_values, rmse_values, "o-", color="#2196F3", linewidth=2, markersize=8)
    ax.set_xlabel("Number of Sparse Anchors (N)", fontsize=12)
    ax.set_ylabel("RMSE (meters)", fontsize=12)
    ax.set_title("Sparsity Sensitivity Analysis", fontsize=14)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Visualizer] Saved sparsity plot → {save_path}")

    plt.show()


def view_pointcloud(pcd):
    """
    Launch the Open3D interactive viewer for a point cloud.
    Only works in a desktop environment with display access.
    """
    import open3d as o3d

    print("[Visualizer] Launching Open3D viewer ... (close window to continue)")
    o3d.visualization.draw_geometries(
        [pcd],
        window_name="Single-View Fusion — 3D Point Cloud",
        width=1280,
        height=720,
    )
