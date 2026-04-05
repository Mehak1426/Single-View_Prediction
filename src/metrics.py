"""
Evaluation Metrics for Depth Estimation.

Computes standard depth prediction metrics by comparing the
aligned (fused) depth map against the ground-truth depth:
  - AbsRel  : Mean absolute relative error
  - RMSE    : Root mean squared error
  - δ<1.25  : Percentage of pixels where max(pred/gt, gt/pred) < 1.25
"""

import numpy as np


def compute_absrel(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> float:
    """
    Absolute Relative Error:  mean( |pred - gt| / gt )

    Args:
        pred  : (H, W) predicted metric depth
        gt    : (H, W) ground-truth metric depth
        valid : (H, W) bool mask for valid pixels

    Returns:
        absrel : float
    """
    return float(np.mean(np.abs(pred[valid] - gt[valid]) / gt[valid]))


def compute_rmse(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> float:
    """
    Root Mean Squared Error:  sqrt( mean( (pred - gt)^2 ) )
    """
    return float(np.sqrt(np.mean((pred[valid] - gt[valid]) ** 2)))


def compute_delta(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray, threshold: float = 1.25) -> float:
    """
    Threshold Accuracy:  % of pixels where max(pred/gt, gt/pred) < threshold.

    The standard thresholds in depth estimation are δ < 1.25, 1.25², 1.25³.
    """
    ratio = np.maximum(pred[valid] / gt[valid], gt[valid] / pred[valid])
    return float(np.mean(ratio < threshold))


def evaluate(pred: np.ndarray, gt: np.ndarray) -> dict:
    """
    Run all depth metrics.

    Args:
        pred : (H, W) aligned metric depth
        gt   : (H, W) ground-truth metric depth

    Returns:
        dict with keys:  absrel, rmse, delta_1.25, delta_1.25^2, delta_1.25^3
    """
    # valid mask: positive and finite in both maps
    valid = (gt > 0) & np.isfinite(gt) & (pred > 0) & np.isfinite(pred)

    if valid.sum() == 0:
        print("[Metrics] WARNING: no valid pixels for evaluation")
        return {"absrel": float("nan"), "rmse": float("nan"),
                "delta_1.25": float("nan"), "delta_1.25^2": float("nan"),
                "delta_1.25^3": float("nan")}

    results = {
        "absrel":       compute_absrel(pred, gt, valid),
        "rmse":         compute_rmse(pred, gt, valid),
        "delta_1.25":   compute_delta(pred, gt, valid, 1.25),
        "delta_1.25^2": compute_delta(pred, gt, valid, 1.25 ** 2),
        "delta_1.25^3": compute_delta(pred, gt, valid, 1.25 ** 3),
    }

    return results


def print_metrics(results: dict):
    """Pretty-print evaluation results."""
    print("\n" + "=" * 50)
    print("  Depth Evaluation Results")
    print("=" * 50)
    print(f"  AbsRel       : {results['absrel']:.4f}")
    print(f"  RMSE         : {results['rmse']:.4f} m")
    print(f"  δ < 1.25     : {results['delta_1.25']:.4f}")
    print(f"  δ < 1.25²    : {results['delta_1.25^2']:.4f}")
    print(f"  δ < 1.25³    : {results['delta_1.25^3']:.4f}")
    print("=" * 50 + "\n")
