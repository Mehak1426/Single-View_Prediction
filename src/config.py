"""
Default configuration for the Single-View Fusion Engine.
All hyperparameters and paths are centralized here.
"""

CONFIG = {
    # ── Model ──────────────────────────────────────────────
    "model_name": "depth-anything/Depth-Anything-V2-Large-hf",
    "device": "cuda",            # "cuda" or "cpu"

    # ── Dataset ────────────────────────────────────────────
    "data_path": "data/nyu_depth_v2.h5",   # path to the HDF5 file
    "image_height": 480,
    "image_width": 640,

    # ── NYU Depth V2 Camera Intrinsics (default) ──────────
    # focal lengths and principal point for 640×480
    "fx": 518.8579,
    "fy": 519.4696,
    "cx": 325.5824,
    "cy": 253.7362,

    # ── Sparse Anchor Sampling ─────────────────────────────
    "num_anchors": 100,           # N sparse points from GT

    # ── RANSAC Alignment ───────────────────────────────────
    "ransac_iterations": 1000,
    "ransac_threshold": 0.1,      # inlier threshold (meters)
    "ransac_min_samples": 10,     # minimum points for fitting

    # ── Sparsity Sweep ─────────────────────────────────────
    "sparsity_values": [5, 10, 50, 100, 500],
    "sparsity_trials": 5,         # average over multiple runs

    # ── Output ─────────────────────────────────────────────
    "output_dir": "outputs",
    "ply_filename": "pointcloud.ply",
}
