"""
Export a single video tensor to numbered image frames on disk.

Supports BTCHW [B, T, C, H, W] or BCTHW [B, C, T, H, W] (set video_format or auto-detect).
"""
import os
from typing import Optional

import torch
from PIL import Image


def save_video_tensor_to_frames(
    video: torch.Tensor,
    save_dir: str,
    video_format: Optional[str] = None,
) -> None:
    """
    Save video tensor to frame images.

    Args:
        video (torch.Tensor): Video, shape [B, T, C, H, W] (BTCHW) or [B, C, T, H, W] (BCTHW).
        save_dir (str): Directory to save frames.
        video_format (Optional[str]): "BTCHW" or "BCTHW". If None, auto-detects
            based on dimensions.

    Returns:
        None: Saves frames to the specified directory.
    """
    os.makedirs(save_dir, exist_ok=True)
    video = video.detach().cpu().float()

    # Auto-detect format
    if video_format is None:
        _, d1, d2, _, _ = video.shape
        video_format = "BCTHW" if (d1 <= 4 and d2 > d1) else "BTCHW"

    # Convert to THWC: [T, H, W, C]
    if video_format == "BCTHW":
        video = video[0].permute(1, 2, 3, 0)  # BCTHW -> [C,T,H,W] -> [T,H,W,C]
    else:  # BTCHW
        video = video[0].permute(0, 2, 3, 1)  # BTCHW -> [T,C,H,W] -> [T,H,W,C]

    # Normalize to [0, 1] using min-max, then scale to [0, 255]
    v_min, v_max = video.min(), video.max()
    if v_max > v_min:
        video = (video - v_min) / (v_max - v_min)
    video = (video * 255).to(torch.uint8).numpy()

    # Save frames
    for i, frame in enumerate(video):
        Image.fromarray(frame).save(
            os.path.join(save_dir, f"frame{i:03d}.jpg"), quality=95
        )
