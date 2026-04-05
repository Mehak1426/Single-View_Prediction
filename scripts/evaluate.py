#!/usr/bin/env python3
"""
evaluate.py — Batch Evaluation Script.

Runs the full pipeline on multiple images from the NYU Depth V2 dataset
and reports aggregated depth evaluation metrics in a table.

Usage:
    python scripts/evaluate.py --num-images 50 --anchors 100
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
from src.metrics import evaluate


def parse_args():
    parser = argparse.ArgumentParser(description="Batch Depth Evaluation")
    parser.add_argument("--num-images", type=int, default=50,
                        help="Number of images to evaluate (default: 50)")
    parser.add_argument("--anchors", type=int, default=CONFIG["num_anchors"],
                        help="Sparse anchors per image")
    parser.add_argument("--data", type=str, default=CONFIG["data_path"],
                        help="Path to HDF5 file")
    parser.add_argument("--device", type=str, default=CONFIG["device"])
    parser.add_argument("--output", type=str, default=CONFIG["output_dir"])
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    # load dataset & model
    dataset = NYUDepthV2Dataset(h5_path=args.data)
    estimator = DepthEstimator(device=args.device)
    sampler = SparseAnchorSampler(num_anchors=args.anchors)
    aligner = RANSACAligner()

    num_images = min(args.num_images, len(dataset))
    all_results = []

    print(f"\n[Evaluate] Running on {num_images} images, N={args.anchors} anchors\n")

    for idx in tqdm(range(num_images), desc="Evaluating"):
        rgb, depth_gt_tensor, K = dataset[idx]

        # convert to numpy
        rgb_np = (rgb.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        depth_gt = depth_gt_tensor.squeeze(0).numpy()

        # predict
        depth_pred = estimator.predict(rgb_np)

        # sample anchors & align
        coords, metric_values = sampler.sample(depth_gt)
        pred_at_anchors = depth_pred[coords[:, 0], coords[:, 1]]

        try:
            scale, shift, _ = aligner.fit(pred_at_anchors, metric_values)
            depth_aligned = RANSACAligner.align(depth_pred, scale, shift)
            results = evaluate(depth_aligned, depth_gt)
            all_results.append(results)
        except Exception as e:
            print(f"  [WARN] Skipping idx {idx}: {e}")
            continue

    # aggregate
    if not all_results:
        print("[Evaluate] No valid results!")
        return

    keys = all_results[0].keys()
    avg = {k: np.mean([r[k] for r in all_results]) for k in keys}
    std = {k: np.std([r[k] for r in all_results]) for k in keys}

    # print table
    print("\n" + "=" * 60)
    print(f"  Aggregated Results ({len(all_results)} images, N={args.anchors})")
    print("=" * 60)
    print(f"  {'Metric':<15} {'Mean':>10} {'Std':>10}")
    print("-" * 60)
    for k in keys:
        print(f"  {k:<15} {avg[k]:>10.4f} {std[k]:>10.4f}")
    print("=" * 60)

    # save to file
    results_path = os.path.join(args.output, "eval_results.txt")
    with open(results_path, "w") as f:
        f.write(f"Evaluation: {len(all_results)} images, N={args.anchors}\n\n")
        for k in keys:
            f.write(f"{k}: {avg[k]:.4f} ± {std[k]:.4f}\n")
    print(f"\n[Evaluate] Results saved → {results_path}")


if __name__ == "__main__":
    main()
