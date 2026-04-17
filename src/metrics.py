"""
Evaluation Metrics for Depth Estimation.

Computes standard depth prediction metrics by comparing the
aligned (fused) depth map against the ground-truth depth:
  - AbsRel  : Mean absolute relative error
  - SqRel   : Mean squared relative error
  - RMSE    : Root mean squared error
  - RMSElog : Root mean squared error of log depth
  - MAE     : Mean absolute error
  - δ<1.25  : Percentage of pixels where max(pred/gt, gt/pred) < 1.25
  - SILog   : Scale-invariant logarithmic error
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


def compute_sqrel(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> float:
    """
    Squared Relative Error:  mean( (pred - gt)^2 / gt )
    """
    return float(np.mean(((pred[valid] - gt[valid]) ** 2) / gt[valid]))


def compute_rmse(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> float:
    """
    Root Mean Squared Error:  sqrt( mean( (pred - gt)^2 ) )
    """
    return float(np.sqrt(np.mean((pred[valid] - gt[valid]) ** 2)))


def compute_rmse_log(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> float:
    """
    Root Mean Squared Logarithmic Error:
        sqrt( mean( (log(pred) - log(gt))^2 ) )
    """
    # Additional safety: ensure positive values for log
    log_valid = valid & (pred > 0) & (gt > 0)
    if log_valid.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((np.log(pred[log_valid]) - np.log(gt[log_valid])) ** 2)))


def compute_mae(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> float:
    """
    Mean Absolute Error:  mean( |pred - gt| )
    """
    return float(np.mean(np.abs(pred[valid] - gt[valid])))


def compute_silog(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> float:
    """
    Scale-Invariant Logarithmic Error (Eigen et al., 2014):
        sqrt( mean(d^2) - (mean(d))^2 )  where  d = log(pred) - log(gt)
    Measures depth quality independent of global scale.
    """
    log_valid = valid & (pred > 0) & (gt > 0)
    if log_valid.sum() == 0:
        return float("nan")
    d = np.log(pred[log_valid]) - np.log(gt[log_valid])
    return float(np.sqrt(np.mean(d ** 2) - (np.mean(d)) ** 2) * 100.0)


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
        dict with keys:  absrel, sqrel, rmse, rmse_log, mae, silog,
                          delta_1.25, delta_1.25^2, delta_1.25^3
    """
    # valid mask: positive and finite in both maps
    valid = (gt > 0) & np.isfinite(gt) & (pred > 0) & np.isfinite(pred)

    if valid.sum() == 0:
        print("[Metrics] WARNING: no valid pixels for evaluation")
        return {k: float("nan") for k in [
            "absrel", "sqrel", "rmse", "rmse_log", "mae", "silog",
            "delta_1.25", "delta_1.25^2", "delta_1.25^3"
        ]}

    results = {
        "absrel":       compute_absrel(pred, gt, valid),
        "sqrel":        compute_sqrel(pred, gt, valid),
        "rmse":         compute_rmse(pred, gt, valid),
        "rmse_log":     compute_rmse_log(pred, gt, valid),
        "mae":          compute_mae(pred, gt, valid),
        "silog":        compute_silog(pred, gt, valid),
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
    print(f"  SqRel        : {results['sqrel']:.6f}")
    print(f"  RMSE         : {results['rmse']:.4f} m")
    print(f"  RMSE(log)    : {results['rmse_log']:.4f}")
    print(f"  MAE          : {results['mae']:.4f} m")
    print(f"  SILog        : {results['silog']:.4f}")
    print(f"  δ < 1.25     : {results['delta_1.25']:.4f}")
    print(f"  δ < 1.25²    : {results['delta_1.25^2']:.4f}")
    print(f"  δ < 1.25³    : {results['delta_1.25^3']:.4f}")
    print("=" * 50 + "\n")
