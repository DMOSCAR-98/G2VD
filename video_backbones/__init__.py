"""
Video backbones for feature extraction from video batches.

Inputs to each backbone are BTCHW tensors, shape [B, T, C, H, W], unless a class docstring
states stricter T or H, W constraints.

Available models:
- XCLIPVideoBackbone: Video backbone based on X-CLIP
- CLIPVideoBackbone: Video backbone based on frame-wise CLIP
- DeMambaXCLIPVideoBackbone: Video backbone based on DeMamba-XCLIP
- DeMambaCLIPVideoBackbone: Video backbone based on DeMamba-CLIP
"""
from .xclip_video_backbone import XCLIPVideoBackbone
from .clip_video_backbone import CLIPVideoBackbone
from .demamba_xclip_video_backbone import DeMambaXCLIPVideoBackbone
from .demamba_clip_video_backbone import DeMambaCLIPVideoBackbone


__all__ = [
    "XCLIPVideoBackbone",
    "CLIPVideoBackbone",
    "DeMambaXCLIPVideoBackbone",
    "DeMambaCLIPVideoBackbone",
]
