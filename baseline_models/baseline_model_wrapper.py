"""
Baseline Model Wrapper for AI-Generated Video Detection

Provides a unified wrapper for baseline models to integrate with the current project.
All baseline models are wrapped to have consistent input/output format compatible
with VideoDetector interface.
"""
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .f3net import Det_F3_Net
from .npr import resnet50_npr
from .stil import Det_STIL
from .tall import TALL_SWIN
from .ftcn import ViT_B_FTCN
from .mintime import ViT_B_MINTIME
from .timesformer import TimeSformer
from .videomae import VideoMAE
from .vivit import ViViT


class BaselineModelWrapper(nn.Module):
    """
    Unified wrapper for baseline models to integrate with the current project.

    This wrapper ensures all baseline models have consistent interface:
    - Input: pixel_values, shape [B, T, C, H, W] (BTCHW).
    - Output: dict with "logits" only, shape [B, 1].

    Note: Baseline models do not provide video-level representations or CD outputs, so:
    - "video_level_reps" is not included in outputs
    - Only "logits" is returned; no causal_logits, mask, etc.
    - forward() accepts **kwargs for interface compatibility (e.g. same signature
      as VideoDetector) but ignores them.

    Args:
        model_name (str): Key into MODEL_MAPPING; must match exactly. Options:
            "f3net", "npr", "stil", "tall", "ftcn", "mintime",
            "timesformer", "videomae", "vivit".
        model_params (Optional[Dict[str, Any]]): Omit or set null in YAML to use
            each baseline model's default constructor. Use a non-empty dict only
            when a baseline needs explicit overrides. Prefer null over an empty
            mapping so configs stay explicit.

    Attributes:
        model: The wrapped baseline model.
        model_name: Name of the baseline model.
        _init_params: Saved initialization parameters for checkpoint loading.

    Examples:
        >>> # Initialize FTCN baseline model (default parameters)
        >>> wrapper = BaselineModelWrapper(model_name="ftcn")
        >>>
        >>> # Forward pass (compatible with VideoDetector interface)
        >>> outputs = wrapper(pixel_values=video_tensor)
        >>> logits = outputs["logits"]  # [B, 1]
    """

    # Model mapping: model_name -> model class.
    MODEL_MAPPING = {
        "f3net": Det_F3_Net,
        "npr": resnet50_npr,
        "stil": Det_STIL,
        "tall": TALL_SWIN,
        "ftcn": ViT_B_FTCN,
        "mintime": ViT_B_MINTIME,
        "timesformer": TimeSformer,
        "videomae": VideoMAE,
        "vivit": ViViT,
    }

    def __init__(
        self,
        model_name: str,
        model_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        if model_name not in self.MODEL_MAPPING:
            raise ValueError(
                f"Unsupported baseline model: {model_name}. "
                f"Please choose from: {list(self.MODEL_MAPPING.keys())}"
            )

        # Save initialization parameters (for checkpoint saving)
        self._init_params = {
            "model_name": model_name,
            "model_params": model_params,
        }
        self.model_name = model_name

        # Initialize baseline model
        model_class = self.MODEL_MAPPING[model_name]
        if model_params is None:
            self.model = model_class()
        else:
            self.model = model_class(**model_params)

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the baseline model.

        Converts baseline model output to the unified format expected by
        the training/evaluation pipeline.

        Accepts **kwargs for interface compatibility with the training pipeline;
        they are ignored.

        Args:
            pixel_values (torch.Tensor, optional): Video input, shape [B, T, C, H, W] (BTCHW).
            **kwargs: Ignored; for compatibility with detector interface.

        Returns:
            Dict[str, torch.Tensor]: Dict with "logits" only, shape [B, 1].
            No video_level_reps or CD-related keys.
        """
        logits = self.model(pixel_values)
        return {"logits": logits}

    def get_init_params(self) -> Dict[str, Any]:
        """
        Get initialization parameters for checkpoint saving.

        Returns:
            Dict[str, Any]: Dictionary containing model initialization parameters.
        """
        return self._init_params.copy()
