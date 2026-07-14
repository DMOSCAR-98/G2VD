"""
Project utilities: data paths, augmentation, LoRA, metrics, checkpoints, and config instantiation.

Docstrings follow Google style in English; see .cursor/rules/python-docstrings.mdc.
"""
from .apply_augmentation import apply_augmentation
from .apply_lora import apply_lora
from .convert_labeled_video_paths import convert_labeled_video_paths
from .data_sampler import data_sampler
from .dp_utils import get_inner_model
from .hinge_loss import hinge_loss
from .instantiate_from_config import (
    get_obj_from_str,
    instantiate_from_config,
)
from .dynamic_parameter_scheduler import dynamic_parameter_scheduler
from .load_labeled_video_paths import load_labeled_video_paths
from .load_metadata_json_paths import load_metadata_json_paths
from .save_model_checkpoint import save_model_checkpoint
from .set_global_seed import set_global_seed
from .metrics_accumulator import metrics_accumulator
from .model_info import linear_layer_info, trainable_parameter_info
from .save_video_tensor_to_frames import save_video_tensor_to_frames
from .vanilla_loss import vanilla_loss
from .visualize_tsne import visualize_tsne


__all__ = [
    "apply_augmentation",
    "apply_lora",
    "convert_labeled_video_paths",
    "data_sampler",
    "get_inner_model",
    "get_obj_from_str",
    "hinge_loss",
    "instantiate_from_config",
    "linear_layer_info",
    "dynamic_parameter_scheduler",
    "load_labeled_video_paths",
    "load_metadata_json_paths",
    "metrics_accumulator",
    "save_model_checkpoint",
    "save_video_tensor_to_frames",
    "set_global_seed",
    "trainable_parameter_info",
    "vanilla_loss",
    "visualize_tsne",
]
