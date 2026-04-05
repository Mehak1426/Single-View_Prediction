#!/usr/bin/env python3
"""
run_pipeline.py — End-to-end Single-View Fusion Pipeline.

Usage:
    python scripts/run_pipeline.py --index 0 --anchors 100

This script:
  1. Loads an image from the NYU Depth V2 dataset
  2. Runs Depth Anything V2 to get a relative depth map
  3. Samples sparse metric anchors from the ground-truth depth
  4. Aligns the prediction to metric scale via RANSAC
  5. Projects the fused depth into a 3D point cloud (.ply)
  6. Saves comparison plots and prints evaluation metrics
"""

import os
import sys
import argparse

# allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.config import CONFIG
from src.dataloader import NYUDepthV2Dataset, get_sample
from src.depth_estimator import DepthEstimator
from src.aligner import SparseAnchorSampler, RANSACAligner
from src.projector import PointCloudProjector
from src.visualizer import plot_depth_comparison, plot_error_map
from src.metrics import evaluate, print_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Single-View Fusion Pipeline")
    parser.add_argument("--index", type=int, default=0,
                        help="Image index in the dataset (default: 0)")
    parser.add_argument("--anchors", type=int, default=CONFIG["num_anchors"],
                        help=f"Number of sparse anchors (default: {CONFIG['num_anchors']})")
    parser.add_argument("--data", type=str, default=CONFIG["data_path"],
                        help="Path to the NYU Depth V2 .h5 file")
    parser.add_argument("--output", type=str, default=CONFIG["output_dir"],
                        help="Output directory")
    parser.add_argument("--device", type=str, default=CONFIG["device"],
                        help="Device: cuda or cpu")
    parser.add_argument("--no-viz", action="store_true",
                        help="Disable matplotlib visualization")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    # ── Step 1: Load Data ──────────────────────────────────
    print("\n[Pipeline] Step 1/5 — Loading data ...")
    rgb_np, depth_gt, K = get_sample(idx=args.index, h5_path=args.data)
    print(f"  Image shape : {rgb_np.shape}")
    print(f"  Depth range : [{depth_gt.min():.2f}, {depth_gt.max():.2f}] m")

    # ── Step 2: Monocular Depth Estimation ─────────────────
    print("\n[Pipeline] Step 2/5 — Running Depth Anything V2 ...")
    estimator = DepthEstimator(device=args.device)
    depth_pred = estimator.predict(rgb_np)
    print(f"  Predicted depth range: [{depth_pred.min():.2f}, {depth_pred.max():.2f}]")

    # ── Step 3: Sparse Anchor Sampling + RANSAC Alignment ──
    print(f"\n[Pipeline] Step 3/5 — Sampling {args.anchors} anchors & aligning ...")
    sampler = SparseAnchorSampler(num_anchors=args.anchors)
    coords, metric_values = sampler.sample(depth_gt)

    # get predicted depth at anchor locations
    pred_at_anchors = depth_pred[coords[:, 0], coords[:, 1]]

    aligner = RANSACAligner()
    scale, shift, inlier_mask = aligner.fit(pred_at_anchors, metric_values)
    depth_aligned = RANSACAligner.align(depth_pred, scale, shift)

    # ── Step 4: 3D Point Cloud Projection ──────────────────
    print("\n[Pipeline] Step 4/5 — Projecting to 3D point cloud ...")
    projector = PointCloudProjector(intrinsics=K)
    ply_path = os.path.join(args.output, f"pointcloud_idx{args.index}.ply")
    projector.export_ply(depth_aligned, rgb=rgb_np, filepath=ply_path)

    # ── Step 5: Evaluation ─────────────────────────────────
    print("\n[Pipeline] Step 5/5 — Computing metrics ...")
    results = evaluate(depth_aligned, depth_gt)
    print_metrics(results)

    # ── Visualizations ─────────────────────────────────────
    if not args.no_viz:
        fig_path = os.path.join(args.output, f"comparison_idx{args.index}.png")
        plot_depth_comparison(
            rgb_np, depth_pred, depth_aligned, depth_gt, save_path=fig_path
        )
        err_path = os.path.join(args.output, f"error_map_idx{args.index}.png")
        plot_error_map(depth_aligned, depth_gt, save_path=err_path)

    print("\n[Pipeline] Done! Outputs saved to:", args.output)


if __name__ == "__main__":
    main()
