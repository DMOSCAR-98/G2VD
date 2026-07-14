"""
Checkpoint saving utility.

This project supports both regular fine-tuning and LoRA fine-tuning (via PEFT).
For LoRA models, we save *only* the merged full-weights checkpoint so it can be
loaded later using standard PyTorch load_state_dict() without requiring PEFT.
"""

from typing import Any, Dict, Optional

import copy
import torch
import torch.nn as nn


def _merge_lora_weights(model: nn.Module) -> nn.Module:
    """
    Recursively merge LoRA adapter weights into the base weights in-place.

    PEFT LoRA modules expose merge_and_unload(). After merging, the returned
    module is a regular (non-PEFT) module with updated weights.
    """
    # If the top-level module itself is a PEFT module, merge it first.
    if hasattr(model, "merge_and_unload"):
        model = model.merge_and_unload()

    for name, module in model.named_children():
        setattr(model, name, _merge_lora_weights(module))

    return model


def save_model_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    is_lora: bool = False,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save a model checkpoint.

    - If is_lora is True: merge LoRA weights into base weights and save the
      *merged* full model "state_dict" (no adapter-only checkpoint).
    - If is_lora is False: save the model "state_dict" directly (default).

    When the model is wrapped with DataParallel, the caller should pass the
    inner module (e.g. via get_inner_model(model)) so that saved state_dict
    keys do not contain the "module." prefix, ensuring compatibility when
    loading on single-GPU or different GPU counts.

    Args:
        model (nn.Module): PyTorch module to save. Can be any nn.Module instance
            (including wrapper modules or composite models). We always save the
            whole module's "state_dict", which includes all parameters from the
            module and its submodules.
        checkpoint_path (str): Output checkpoint file path.
        is_lora (bool): Whether the model is trained with LoRA. Defaults to False
            so that the common case (non-LoRA fine-tuning) can omit this argument.
        extra (Optional[Dict[str, Any]]): Optional extra fields to include in the
            checkpoint dict. If not provided, only {"state_dict": ...} is saved.
    """
    payload: Dict[str, Any] = {}
    if extra:
        payload.update(extra)

    if is_lora:
        # merge_and_unload() mutates modules; deepcopy to avoid touching the
        # in-training model instance.
        merged_model = copy.deepcopy(model).cpu()
        _merge_lora_weights(merged_model)
        payload["state_dict"] = merged_model.state_dict()
        payload["merged_lora"] = True
    else:
        payload["state_dict"] = model.state_dict()
        payload["merged_lora"] = False

    torch.save(payload, checkpoint_path)
