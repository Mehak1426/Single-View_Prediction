"""
NYU Depth V2 Data Loader.
Loads RGB images, ground-truth depth maps, and camera intrinsics
from the folder-based dataset with CSV manifests.

Expected layout:
    nyu_data/
        data/
            nyu2_train.csv        # each line: image_path,depth_path
            nyu2_test.csv
            nyu2_train/           # PNG image/depth pairs
            nyu2_test/
"""

import os
import csv
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from src.config import CONFIG


class NYUDepthV2Dataset(Dataset):
    """
    Wraps the NYU Depth V2 folder-based dataset into a PyTorch Dataset.

    The dataset uses CSV manifests (nyu2_train.csv / nyu2_test.csv) where
    each line contains:
        image_path,depth_path

    Paths in the CSV are relative to the dataset root directory.

    Each __getitem__ returns:
        rgb        : torch.FloatTensor  (3, H, W)  in [0, 1]
        depth_gt   : torch.FloatTensor  (1, H, W)  in meters
        intrinsics : torch.FloatTensor  (3, 3)      camera matrix K
    """

    def __init__(self, data_dir: str = None, split: str = "train", transform=None):
        """
        Args:
            data_dir  : path to the dataset root (the folder containing the
                        CSV files and the data/ subfolder).
                        Defaults to CONFIG["data_dir"].
            split     : "train" or "test"
            transform : optional torchvision transform for the RGB image
        """
        self.data_dir = data_dir or CONFIG["data_dir"]
        self.split = split
        self.transform = transform

        # build the intrinsic matrix once
        self.K = torch.tensor([
            [CONFIG["fx"],  0.0,          CONFIG["cx"]],
            [0.0,           CONFIG["fy"], CONFIG["cy"]],
            [0.0,           0.0,          1.0         ],
        ], dtype=torch.float32)

        # read the CSV manifest
        csv_name = f"nyu2_{split}.csv"
        csv_path = os.path.join(self.data_dir, csv_name)
        if not os.path.isfile(csv_path):
            # fallback: CSV may be inside a data/ subfolder
            csv_path = os.path.join(self.data_dir, "data", csv_name)
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(
                f"CSV manifest not found: {csv_name}\n"
                f"Searched in: {self.data_dir} and {os.path.join(self.data_dir, 'data')}"
            )

        self.pairs = []
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2:
                    continue
                img_path = row[0].strip()
                depth_path = row[1].strip()
                self.pairs.append((img_path, depth_path))

        if not self.pairs:
            raise RuntimeError(f"No image/depth pairs found in {csv_path}")

        self.length = len(self.pairs)

    def __len__(self) -> int:
        return self.length

    def _resolve_path(self, rel_path: str) -> str:
        """Resolve a path from the CSV relative to data_dir."""
        full = os.path.join(self.data_dir, rel_path)
        if os.path.isfile(full):
            return full
        # some CSVs use paths relative to the data/ subfolder
        alt = os.path.join(self.data_dir, "data", rel_path)
        if os.path.isfile(alt):
            return alt
        # try treating the path as-is (absolute or relative to cwd)
        if os.path.isfile(rel_path):
            return rel_path
        raise FileNotFoundError(
            f"Could not find file: {rel_path}\n"
            f"  Tried: {full}\n"
            f"  Tried: {alt}"
        )

    def __getitem__(self, idx: int):
        img_rel, depth_rel = self.pairs[idx]

        img_path = self._resolve_path(img_rel)
        depth_path = self._resolve_path(depth_rel)

        # load RGB image
        rgb_pil = Image.open(img_path).convert("RGB")
        rgb_np = np.array(rgb_pil, dtype=np.float32)           # (H, W, 3)

        # load depth map (16-bit or 8-bit PNG → metres)
        depth_pil = Image.open(depth_path)
        depth_np = np.array(depth_pil, dtype=np.float32)       # (H, W)

        # NYU depth PNGs are typically stored as uint16 with a scale of
        # 1/5000 (depth_in_metres = pixel_value / 5000).  Handle both
        # raw-metre floats and the scaled-integer convention.
        if depth_np.max() > 20.0:
            depth_np = depth_np / 5000.0

        # to tensors
        rgb = torch.from_numpy(rgb_np).permute(2, 0, 1) / 255.0   # (3, H, W)
        depth_gt = torch.from_numpy(depth_np).float()              # (H, W)

        if self.transform is not None:
            rgb = self.transform(rgb)

        depth_gt = depth_gt.unsqueeze(0)     # (1, H, W)

        return rgb, depth_gt, self.K


def get_sample(idx: int = 0, data_dir: str = None, split: str = "train"):
    """
    Quick helper to grab a single sample as numpy arrays.
    Useful for scripts and debugging.

    Returns:
        rgb_np     : np.ndarray (H, W, 3) uint8
        depth_np   : np.ndarray (H, W)    float32, metres
        K          : np.ndarray (3, 3)     float64
    """
    dataset = NYUDepthV2Dataset(data_dir=data_dir, split=split)
    rgb, depth_gt, K = dataset[idx]

    rgb_np = (rgb.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    depth_np = depth_gt.squeeze(0).numpy()
    K_np = K.numpy()

    return rgb_np, depth_np, K_np
