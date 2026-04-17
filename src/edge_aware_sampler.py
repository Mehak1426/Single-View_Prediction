import numpy as np
from src.config import CONFIG

class EdgeAwareAnchorSampler:
    """
    Novelty: Geometry-Aware Sparse Anchor Sampling.
    Computes depth gradients to avoid sampling points near object boundaries.
    By forcing the sampler to select anchors only on "flat" surfaces, 
    we reduce the likelihood of RANSAC pulling outliers caused by 
    monocular edge-bleeding artifacts.
    """

    def __init__(self, num_anchors: int = None, seed: int = None, edge_threshold: float = 0.3):
        self.num_anchors = num_anchors or CONFIG.get("num_anchors", 100)
        self.rng = np.random.default_rng(seed)
        
        # threshold in meters: gradients > this value are considered edges
        self.edge_threshold = edge_threshold 

    def sample(self, depth_gt: np.ndarray):
        """
        Pick N valid pixels from the GT depth map while actively 
        avoiding areas with high depth gradients.
        """
        # 1. Base mask: only sample from valid (non-zero, finite) pixels
        valid_mask = (depth_gt > 0) & np.isfinite(depth_gt)

        # 2. Compute the spatial gradient to find edges
        # We copy and clean up NaN/0s safely for the gradient calculation
        safe_depth = np.copy(depth_gt)
        safe_depth[~valid_mask] = 0.0 
        
        dx, dy = np.gradient(safe_depth)
        gradient_magnitude = np.hypot(dx, dy)

        # 3. Flat Surface Mask: Must be valid AND not near a sharp edge
        flat_mask = valid_mask & (gradient_magnitude < self.edge_threshold)
        
        valid_rows, valid_cols = np.where(flat_mask)

        # Fallback safeguard in case there aren't enough "flat" pixels in a highly cluttered scene
        if len(valid_rows) < self.num_anchors:
            print(f"[Aligner] Warning: Only {len(valid_rows)} flat valid pixels found! Falling back to standard sampling.")
            valid_rows, valid_cols = np.where(valid_mask)

        # 4. Randomly sample from our highly curated "flat" points
        n = min(self.num_anchors, len(valid_rows))
        indices = self.rng.choice(len(valid_rows), size=n, replace=False)

        coords = np.stack([valid_rows[indices], valid_cols[indices]], axis=1)
        values = depth_gt[coords[:, 0], coords[:, 1]]

        return coords, values
