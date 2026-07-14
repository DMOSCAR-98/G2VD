"""
Baseline detectors for ai-generated video detection.

All are used through BaselineModelWrapper with BTCHW video tensors [B, T, C, H, W].

"""
from .f3net import Det_F3_Net
from .npr import resnet50_npr
from .stil import Det_STIL
from .tall import TALL_SWIN
from .ftcn import ViT_B_FTCN
from .mintime import ViT_B_MINTIME
from .timesformer import TimeSformer
from .videomae import VideoMAE
from .vivit import ViViT

# Import wrapper for unified interface
from .baseline_model_wrapper import BaselineModelWrapper


__all__ = [
    "Det_F3_Net",
    "resnet50_npr",
    "Det_STIL",
    "TALL_SWIN",
    "ViT_B_FTCN",
    "ViT_B_MINTIME",
    "TimeSformer",
    "VideoMAE",
    "ViViT",
    "BaselineModelWrapper",
]
