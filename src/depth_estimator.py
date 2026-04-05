"""
Depth Anything V2 — Monocular Depth Estimator.

Loads the Depth Anything V2 Large model via HuggingFace Transformers
and exposes a simple predict() interface that returns a dense depth map.

Reference: https://arxiv.org/abs/2406.09414
"""

import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from src.config import CONFIG


class DepthEstimator:
    """
    Wrapper around Depth Anything V2 for monocular depth prediction.

    The model produces "relative" (affine-invariant) depth — i.e. the
    per-pixel values are consistent in ordering and relative scale, but
    they are NOT in metric units. We fix that in the Aligner stage.
    """

    def __init__(self, model_name: str = None, device: str = None):
        """
        Args:
            model_name : HuggingFace model identifier
            device     : 'cuda' or 'cpu'
        """
        self.device = device or CONFIG["device"]
        model_name = model_name or CONFIG["model_name"]

        print(f"[DepthEstimator] Loading {model_name} ...")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        print(f"[DepthEstimator] Model ready on {self.device}")

    @torch.no_grad()
    def predict(self, image) -> np.ndarray:
        """
        Run monocular depth estimation on a single image.

        Args:
            image : PIL.Image, np.ndarray (H, W, 3) uint8,
                     or torch.Tensor (3, H, W) in [0, 1]

        Returns:
            depth_pred : np.ndarray (H, W) float32
                         Relative depth map (NOT metric).
        """
        # handle different input types
        if isinstance(image, torch.Tensor):
            # (3, H, W) float [0,1] → PIL
            image_np = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            image_pil = Image.fromarray(image_np)
        elif isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image)
        else:
            image_pil = image  # assume PIL

        # preprocess
        inputs = self.processor(images=image_pil, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # forward pass
        outputs = self.model(**inputs)
        predicted_depth = outputs.predicted_depth  # (1, H', W')

        # upsample to the original image size
        h, w = image_pil.size[1], image_pil.size[0]
        depth_pred = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        return depth_pred.cpu().numpy().astype(np.float32)
