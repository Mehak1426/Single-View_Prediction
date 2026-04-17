#!/usr/bin/env python3
"""
sparsity_analysis.py — Sparsity Sensitivity Study.

Sweeps over different numbers of sparse anchors (N) and measures
how RMSE changes. Produces a publication-ready plot.

Usage:
    python scripts/sparsity_analysis.py --num-images 20
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tqdm import tqdm

from src.config import CONFIG
from src.dataloader import NYUDepthV2Dataset
from src.depth_estimator import DepthEstimator
from src.aligner import SparseAnchorSampler, RANSACAligner
from src.edge_aware_sampler import EdgeAwareAnchorSampler
from src.metrics import compute_rmse
from src.visualizer import plot_sparsity_curve


def parse_args():
    parser = argparse.ArgumentParser(description="Sparsity Sensitivity Analysis")
    parser.add_argument("--num-images", type=int, default=20,
                        help="Number of images to average over (default: 20)")
    parser.add_argument("--trials", type=int, default=CONFIG["sparsity_trials"],
                        help="Random trials per N value (default: 5)")
    parser.add_argument("--data", type=str, default=CONFIG["data_dir"],
                        help="Path to the NYU Depth V2 dataset folder")
    parser.add_argument("--device", type=str, default=CONFIG["device"])
    parser.add_argument("--output", type=str, default=CONFIG["output_dir"])
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    dataset = NYUDepthV2Dataset(data_dir=args.data)
    estimator = DepthEstimator(device=args.device)
    aligner = RANSACAligner()

    n_values = CONFIG["sparsity_values"]  # [5, 10, 50, 100, 500]
    num_images = min(args.num_images, len(dataset))

    print(f"\n[Sparsity] Sweeping N ∈ {n_values}")
    print(f"  Images: {num_images}, Trials per N: {args.trials}\n")

    # precompute predictions for all images (expensive step — do it once)
    print("[Sparsity] Precomputing monocular predictions ...")
    predictions = []
    gt_depths = []
    for idx in tqdm(range(num_images), desc="Predicting"):
        rgb, depth_gt_tensor, _ = dataset[idx]
        rgb_np = (rgb.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        depth_gt = depth_gt_tensor.squeeze(0).numpy()

        depth_pred = estimator.predict(rgb_np)
        predictions.append(depth_pred)
        gt_depths.append(depth_gt)

    # sweep over N values
    mean_rmse = []

    for n in n_values:
        all_rmse = []

        for img_idx in range(num_images):
            depth_pred = predictions[img_idx]
            depth_gt = gt_depths[img_idx]

            for trial in range(args.trials):
                sampler = EdgeAwareAnchorSampler(num_anchors=n, seed=trial * 1000 + img_idx)
                coords, metric_values = sampler.sample(depth_gt)
                pred_at_anchors = depth_pred[coords[:, 0], coords[:, 1]]

                try:
                    scale, shift, _ = aligner.fit(pred_at_anchors, metric_values)
                    depth_aligned = RANSACAligner.align(depth_pred, scale, shift)
                    valid = (depth_gt > 0) & np.isfinite(depth_gt) & (depth_aligned > 0)
                    rmse = compute_rmse(depth_aligned, depth_gt, valid)
                    all_rmse.append(rmse)
                except Exception:
                    continue

        avg_rmse = np.mean(all_rmse) if all_rmse else float("nan")
        mean_rmse.append(avg_rmse)
        print(f"  N={n:>4d}  →  RMSE = {avg_rmse:.4f} m  ({len(all_rmse)} runs)")

    # plot
    plot_path = os.path.join(args.output, "sparsity_sensitivity.png")
    plot_sparsity_curve(n_values, mean_rmse, save_path=plot_path)

    # save raw data
    data_path = os.path.join(args.output, "sparsity_data.txt")
    with open(data_path, "w") as f:
        f.write("N,RMSE\n")
        for n, r in zip(n_values, mean_rmse):
            f.write(f"{n},{r:.6f}\n")
    print(f"\n[Sparsity] Data saved → {data_path}")


if __name__ == "__main__":
    main()
