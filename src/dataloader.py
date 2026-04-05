"""
NYU Depth V2 Data Loader.
Loads RGB images, ground-truth depth maps, and camera intrinsics
from the Kaggle HDF5 archive.

Dataset: https://www.kaggle.com/datasets/soumikrakshit/nyu-depth-v2
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from src.config import CONFIG


class NYUDepthV2Dataset(Dataset):
    """
    Wraps the NYU Depth V2 HDF5 file into a PyTorch Dataset.

    The Kaggle HDF5 stores:
        - images:  (N, 3, H, W) uint8  — RGB images
        - depths:  (N, H, W)   float   — ground-truth depth in meters

    Each __getitem__ returns:
        rgb        : torch.FloatTensor  (3, H, W)  in [0, 1]
        depth_gt   : torch.FloatTensor  (1, H, W)  in meters
        intrinsics : torch.FloatTensor  (3, 3)      camera matrix K
    """

    def __init__(self, h5_path: str = None, transform=None):
        """
        Args:
            h5_path   : path to the .h5 file (defaults to CONFIG value)
            transform : optional torchvision transform for the RGB image
        """
        self.h5_path = h5_path or CONFIG["data_path"]
        self.transform = transform

        # build the intrinsic matrix once
        self.K = torch.tensor([
            [CONFIG["fx"],  0.0,          CONFIG["cx"]],
            [0.0,           CONFIG["fy"], CONFIG["cy"]],
            [0.0,           0.0,          1.0         ],
        ], dtype=torch.float32)

        # open the file briefly to read the dataset length
        with h5py.File(self.h5_path, "r") as f:
            self.length = f["images"].shape[0]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        with h5py.File(self.h5_path, "r") as f:
            # images are stored as (N, 3, H, W) uint8
            rgb = f["images"][idx]            # (3, H, W)
            depth_gt = f["depths"][idx]       # (H, W)

        # convert to float tensors
        rgb = torch.from_numpy(rgb).float() / 255.0        # (3, H, W) [0,1]
        depth_gt = torch.from_numpy(depth_gt).float()      # (H, W)

        # the Kaggle HDF5 stores images as (3, H, W) but they're
        # transposed compared to the original — need to swap axes
        # so we get proper orientation: (3, H, W) → (3, W, H) is wrong,
        # the file actually stores them as (3, 480, 640) already
        # but let's be safe and just ensure correct shape
        if rgb.shape[1] != CONFIG["image_height"]:
            rgb = rgb.permute(0, 2, 1)        # fix transposition if needed
            depth_gt = depth_gt.T

        if self.transform is not None:
            # transform expects PIL or (C, H, W) tensor
            rgb = self.transform(rgb)

        depth_gt = depth_gt.unsqueeze(0)     # (1, H, W)

        return rgb, depth_gt, self.K


def get_sample(idx: int = 0, h5_path: str = None):
    """
    Quick helper to grab a single sample as numpy arrays.
    Useful for scripts and debugging.

    Returns:
        rgb_np     : np.ndarray (H, W, 3) uint8
        depth_np   : np.ndarray (H, W)    float32, meters
        K          : np.ndarray (3, 3)     float64
    """
    dataset = NYUDepthV2Dataset(h5_path=h5_path)
    rgb, depth_gt, K = dataset[idx]

    rgb_np = (rgb.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    depth_np = depth_gt.squeeze(0).numpy()
    K_np = K.numpy()

    return rgb_np, depth_np, K_np
