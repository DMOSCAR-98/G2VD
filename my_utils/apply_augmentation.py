"""
Apply Albumentations transforms to a short video clip with replay consistency.

video_frames uses layout [T, H, W, C] (time-first, channel-last) for albumentations.
"""
import torch

from typing import Any


def apply_augmentation(
    video_frames: torch.Tensor,
    album_transforms: Any,
) -> torch.Tensor:
    """
    Apply augmentation to video frames using album transforms.

    Input/output format is [T, H, W, C] to meet the requirements of
    album_transforms and subsequent processors.

    Args:
        video_frames (torch.Tensor): Frames, shape [T, H, W, C].
        album_transforms (Any): Albumentations transform (supports replay).

    Returns:
        torch.Tensor: Augmented frames, same layout [T, H, W, C].
    """
    frames = [frame.numpy() for frame in video_frames]

    # Apply augmentation to the first frame and extract augmentation parameters
    first_frame_aug = album_transforms(image=frames[0])
    params = first_frame_aug["replay"]

    # Apply the same augmentation parameters to all frames
    transformed_frames = [
        album_transforms.replay(params, image=frame)["image"]
        for frame in frames
    ]

    stacked = torch.stack(
        [torch.from_numpy(f) for f in transformed_frames]
    )
    return stacked
