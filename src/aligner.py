"""
Scale-Shift Alignment via RANSAC.

Solves the affine ambiguity of monocular depth models by fitting:
    d_metric = scale * d_pred + shift

using sparse anchor points sampled from the ground-truth depth map.
RANSAC provides robustness against outliers at object boundaries.
"""

import numpy as np
from sklearn.linear_model import RANSACRegressor, LinearRegression

from src.config import CONFIG


class SparseAnchorSampler:
    """
    Simulates sparse metric anchors by randomly sampling N points
    from the ground-truth depth map. In a real pipeline these would
    come from SfM, LiDAR, or stereo.
    """

    def __init__(self, num_anchors: int = None, seed: int = None):
        self.num_anchors = num_anchors or CONFIG["num_anchors"]
        self.rng = np.random.default_rng(seed)

    def sample(self, depth_gt: np.ndarray):
        """
        Randomly pick N valid pixels from the GT depth map.

        Args:
            depth_gt : (H, W) float32, ground-truth metric depth

        Returns:
            coords  : (N, 2) int array of (row, col) indices
            values  : (N,)   float array of metric depth at those coords
        """
        # only sample from valid (non-zero, finite) pixels
        valid_mask = (depth_gt > 0) & np.isfinite(depth_gt)
        valid_rows, valid_cols = np.where(valid_mask)

        n = min(self.num_anchors, len(valid_rows))
        indices = self.rng.choice(len(valid_rows), size=n, replace=False)

        coords = np.stack([valid_rows[indices], valid_cols[indices]], axis=1)
        values = depth_gt[coords[:, 0], coords[:, 1]]

        return coords, values


class RANSACAligner:
    """
    Fits  d_metric = s * d_pred + t  via RANSAC linear regression.

    RANSAC automatically rejects outlier anchor points where the
    monocular model has large structural errors (e.g. depth
    discontinuities at object edges).
    """

    def __init__(
        self,
        max_trials: int = None,
        residual_threshold: float = None,
        min_samples: int = None,
    ):
        self.max_trials = max_trials or CONFIG["ransac_iterations"]
        self.residual_threshold = residual_threshold or CONFIG["ransac_threshold"]
        self.min_samples = min_samples or CONFIG["ransac_min_samples"]

    def fit(self, d_pred_at_anchors: np.ndarray, d_metric_at_anchors: np.ndarray):
        """
        Solve for (scale, shift) using RANSAC.

        Args:
            d_pred_at_anchors   : (N,) predicted depth at anchor locations
            d_metric_at_anchors : (N,) ground-truth metric depth

        Returns:
            scale       : float
            shift       : float
            inlier_mask : (N,) bool — which anchors were used
        """
        X = d_pred_at_anchors.reshape(-1, 1)
        y = d_metric_at_anchors

        ransac = RANSACRegressor(
            estimator=LinearRegression(),
            max_trials=self.max_trials,
            residual_threshold=self.residual_threshold,
            min_samples=self.min_samples,
            random_state=42,
        )
        ransac.fit(X, y)

        scale = float(ransac.estimator_.coef_[0])
        shift = float(ransac.estimator_.intercept_)
        inlier_mask = ransac.inlier_mask_

        n_inliers = inlier_mask.sum()
        print(f"[Aligner] scale={scale:.4f}, shift={shift:.4f}, "
              f"inliers={n_inliers}/{len(y)}")

        return scale, shift, inlier_mask

    @staticmethod
    def align(d_pred: np.ndarray, scale: float, shift: float) -> np.ndarray:
        """
        Apply s and t to the entire predicted depth map.

        Args:
            d_pred  : (H, W) relative depth from monocular model
            scale   : fitted scale
            shift   : fitted shift

        Returns:
            d_final : (H, W) metric-aligned depth map
        """
        d_final = scale * d_pred + shift
        # clamp to non-negative — negative depth is physically meaningless
        d_final = np.clip(d_final, a_min=0.0, a_max=None)
        return d_final
