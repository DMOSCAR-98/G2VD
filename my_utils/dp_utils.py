"""
DataParallel (DP) utilities for unified single/multi-GPU training.

When the model is wrapped with torch.nn.DataParallel, attribute access
(e.g. replace_classifier, backbone, get_init_params) must target the inner
module. get_inner_model returns the inner module when present, otherwise
the model itself, so the same code path works for both wrapped and unwrapped
models. Use CUDA_VISIBLE_DEVICES to control which GPUs are used; main.py
uses all visible GPUs by default.
"""

import torch


def get_inner_model(model: torch.nn.Module) -> torch.nn.Module:
    """
    Return the inner module when wrapped in DataParallel, else the model.

    Use this when accessing model-specific attributes or methods (e.g.
    replace_classifier, backbone, get_init_params) so that single-GPU and
    multi-GPU (DP) code paths stay unified.

    Args:
        model (torch.nn.Module): Model, possibly wrapped in DataParallel.

    Returns:
        torch.nn.Module: The inner module if model is DataParallel,
            otherwise model unchanged.
    """
    return getattr(model, "module", model)
