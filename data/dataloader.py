"""
Labeled video DataLoaders for training and validation.

Builds PyTorchVideo LabeledVideoDataset pipelines from config. Batches expose video tensors
in BTCHW layout [B, T, C, H, W] after transforms, consistent with the detector pipeline.

Each DataLoader receives a dedicated torch.Generator manual_seed from cfg seed (train uses
seed, val uses seed + 1) so PyTorch derives worker_info seeds from that generator instead
of the global RNG at iter time; together with worker_init_fn, this aligns multi-worker
behavior with the merged cfg seed. LabeledVideoDataset still re-seeds its video-order
Generator in __iter__ via worker_info (see _make_dataloader_worker_init_fn docstring).
"""
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import albumentations
import numpy as np
import torch
from pytorchvideo.data import LabeledVideoDataset, RandomClipSampler
from pytorchvideo.transforms import ApplyTransformToKey, UniformTemporalSubsample
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
    get_worker_info,
)
from torchvision.transforms import Compose, Lambda

from my_utils import apply_augmentation, instantiate_from_config


def _make_dataloader_worker_init_fn(base_seed: int) -> Callable[[int], None]:
    """Return a DataLoader worker_init_fn that seeds random, NumPy, and torch.

    RandomClipSampler uses Python random.uniform; Albumentations uses NumPy/random.
    LabeledVideoDataset re-seeds the video-order torch.Generator in __iter__ using
    worker_info.seed and worker_info.id (PyTorchVideo). DataLoader passes generator=
    seeded from cfg so worker_info is derived from cfg seed, not the global RNG.
    We seed Python/NumPy/torch from worker_info.seed (when available) so each
    DataLoader iteration keeps deterministic but non-repeating epoch-level
    randomness under a fixed experiment seed.

    Args:
        base_seed (int): Same value as cfg seed (entry scripts call
            set_global_seed before create_dataloaders).

    Returns:
        Callable[[int], None]: worker_init_fn(worker_id) for DataLoader.
    """

    def _worker_init_fn(worker_id: int) -> None:
        info = get_worker_info()
        # Prefer DataLoader-provided per-worker/per-iterator seed for stable
        # reproducibility across runs while avoiding identical RNG streams each epoch.
        s = int(info.seed) if info is not None else int(base_seed + worker_id)
        random.seed(s)
        # NumPy expects 32-bit seed range; fold deterministically if needed.
        np.random.seed(s % (2**32 - 1))
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)

    return _worker_init_fn


def _make_dataloader_generator(seed: int) -> torch.Generator:
    """Build a CPU torch.Generator with manual_seed(seed) for DataLoader(generator=).

    PyTorch uses this generator when assigning worker seeds for IterableDataset loaders;
    seeding it from cfg ties worker_info (and thus LabeledVideoDataset video shuffle) to
    the experiment seed instead of whatever the global RNG state is at iter() time.

    Args:
        seed (int): Integer seed; callers use cfg seed or a fixed offset for a second loader.

    Returns:
        torch.Generator: CPU generator ready to pass to DataLoader.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def create_dataloaders(
    cfg: Dict[str, Any],
    train_labeled_video_paths: Optional[List[Tuple[str, Dict[str, Any]]]],
    val_labeled_video_paths: Optional[List[Tuple[str, Dict[str, Any]]]],
) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
    """
    Create training and validation data loaders.

    Args:
        cfg (Dict[str, Any]): Config with the following keys:
            - train_resolution (int): Image resolution for training.
            - train_num_samples (int): Number of sampled frames for training.
            - train_clip_duration (float): Video clip duration for training.
            - train_batch_size (int): Batch size for training.
            - train_num_workers (int): Data loader workers for training.
            - val_resolution (int): Image resolution for validation.
            - val_num_samples (int): Number of sampled frames for validation.
            - val_clip_duration (float): Video clip duration for validation.
            - val_batch_size (int): Batch size for validation.
            - val_num_workers (int): Data loader workers for validation.
            - seed (int, optional): Drives set_global_seed in entry scripts, worker_init_fn
                when num_workers > 0, and DataLoader(generator=) (train: seed, val: seed+1)
                so worker_info and clip/augment RNG align with merged cfg. Defaults to 42
                when omitted (same as entry defaults).
            - test_augmentation (Optional[Dict], optional): If present, used to build
                val augmentation (for test under non-ideal conditions). Format:
                {"transforms": [{"target": "albumentations.TransformName",
                "params": {...}}, ...]}. Each "params" dict must match the
                constructor kwargs of that transform (e.g. Resize: height, width;
                ImageCompression: quality_range, p; GaussNoise: std_range, p;
                GaussianBlur: sigma_limit, p; Normalize: mean, std). When absent,
                val uses Resize + Normalize only.
        train_labeled_video_paths (Optional[List[Tuple[str, Dict[str, Any]]]]):
            List of labeled video paths for training set.
        val_labeled_video_paths (Optional[List[Tuple[str, Dict[str, Any]]]]):
            Labeled video paths for validation set.

    Returns:
        Tuple[Optional[DataLoader], Optional[DataLoader]]: Tuple containing:
            - train_dataloader (Optional[DataLoader]):
                Training data loader, or None if no training data.
            - val_dataloader (Optional[DataLoader]):
                Validation data loader, or None if no validation data.

    Example:
        >>> train_loader, val_loader = create_dataloaders(
        ...     cfg, train_paths, val_paths
        ... )
        >>> if train_loader is not None:
        ...     for epoch in range(num_epochs):
        ...         for batch in train_loader:
        ...             # Training code...
    """
    seed = cfg.get("seed", 42)

    worker_init_fn: Optional[Callable[[int], None]] = None
    if cfg.get("train_num_workers", 0) > 0 or cfg.get("val_num_workers", 0) > 0:
        worker_init_fn = _make_dataloader_worker_init_fn(seed)

    if train_labeled_video_paths:

        # Define training set data augmentation
        max_size_range = list(
            range(cfg["train_resolution"], int(cfg["train_resolution"] * 1.2))
        )
        train_augmentation = albumentations.ReplayCompose([
            albumentations.SmallestMaxSize(max_size=max_size_range),
            albumentations.RandomCrop(
                height=cfg["train_resolution"], width=cfg["train_resolution"]
            ),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.ImageCompression(quality_range=(50, 100), p=0.5),
            albumentations.GaussNoise(std_range=(0.1, 0.3), p=0.3),
            albumentations.GaussianBlur(
                blur_limit=(3, 5), sigma_limit=(0.3, 1.5), p=0.3
            ),
            albumentations.ToGray(p=0.01),
            albumentations.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        # Define training set transformations
        train_transform = Compose([
            ApplyTransformToKey(
                key="video",
                transform=Compose([
                    UniformTemporalSubsample(
                        num_samples=cfg["train_num_samples"]
                    ),  # CTHW
                    Lambda(
                        lambda x: x.clamp(0, 255).to(torch.uint8).permute(
                            1, 2, 3, 0
                        )
                    ),  # THWC
                    Lambda(
                        lambda x: apply_augmentation(x, train_augmentation)
                    ),  # THWC
                    Lambda(lambda x: x.permute(0, 3, 1, 2)),  # TCHW
                ]),
            ),
        ])

        # Create training dataset
        train_dataset = LabeledVideoDataset(
            labeled_video_paths=train_labeled_video_paths,
            clip_sampler=RandomClipSampler(
                clip_duration=cfg["train_clip_duration"]
            ),
            video_sampler=RandomSampler,
            transform=train_transform,
            decode_audio=False,
            decoder="pyav",
        )

        # Create training data loader (prefetch_factor only when num_workers > 0).
        tw = cfg["train_num_workers"]
        train_dl_generator = _make_dataloader_generator(seed)
        train_kw: Dict[str, Any] = {
            "batch_size": cfg["train_batch_size"],
            "num_workers": tw,
            "pin_memory": True,
            "drop_last": False,
            "generator": train_dl_generator,
        }
        if tw > 0:
            train_kw["prefetch_factor"] = 2
            train_kw["worker_init_fn"] = worker_init_fn
        train_dataloader = DataLoader(train_dataset, **train_kw)
    else:
        train_dataloader = None

    if val_labeled_video_paths:

        # Validation set augmentation: default Resize+Normalize; optional
        # test_augmentation for non-ideal test conditions (e.g. noise, compression).
        test_aug_cfg = cfg.get("test_augmentation")
        if test_aug_cfg and test_aug_cfg.get("transforms"):
            transform_list = [
                instantiate_from_config(t)
                for t in test_aug_cfg["transforms"]
            ]
            val_augmentation = albumentations.ReplayCompose(transform_list)
        else:
            val_augmentation = albumentations.ReplayCompose([
                albumentations.Resize(
                    cfg["val_resolution"], cfg["val_resolution"]
                ),
                albumentations.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

        # Define validation set transformations
        val_transform = Compose([
            ApplyTransformToKey(
                key="video",
                transform=Compose([
                    UniformTemporalSubsample(
                        num_samples=cfg["val_num_samples"]
                    ),  # CTHW
                    Lambda(
                        lambda x: x.clamp(0, 255).to(torch.uint8).permute(
                            1, 2, 3, 0
                        )
                    ),  # THWC
                    Lambda(
                        lambda x: apply_augmentation(x, val_augmentation)
                    ),  # THWC
                    Lambda(lambda x: x.permute(0, 3, 1, 2)),  # TCHW
                ]),
            ),
        ])

        # Create validation dataset
        val_dataset = LabeledVideoDataset(
            labeled_video_paths=val_labeled_video_paths,
            clip_sampler=RandomClipSampler(
                clip_duration=cfg["val_clip_duration"]
            ),
            video_sampler=SequentialSampler,
            transform=val_transform,
            decode_audio=False,
            decoder="pyav",
        )

        vw = cfg["val_num_workers"]
        val_dl_generator = _make_dataloader_generator(seed + 1)
        val_kw: Dict[str, Any] = {
            "batch_size": cfg["val_batch_size"],
            "num_workers": vw,
            "pin_memory": True,
            "drop_last": False,
            "generator": val_dl_generator,
        }
        if vw > 0:
            val_kw["prefetch_factor"] = 2
            val_kw["worker_init_fn"] = worker_init_fn
        val_dataloader = DataLoader(val_dataset, **val_kw)
    else:
        val_dataloader = None

    return train_dataloader, val_dataloader
