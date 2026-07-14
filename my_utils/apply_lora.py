"""
Apply PEFT LoRA adapters to a nn.Module (freeze base, inject low-rank weights).

Typical use: video backbones with target_modules such as q_proj, k_proj, v_proj, out_proj.

Project note:
- Verified in this repository for XCLIP LoRA mode on vision_model and mit with
  target_modules ["q_proj", "k_proj", "v_proj", "out_proj"].
- Other target_modules/module structures may be valid in principle but are not
  considered verified here. Keep strict_target_match=True to avoid silent mismatch.
"""
import logging
import torch.nn as nn

from typing import List, Optional, Union


def apply_lora(
    module: nn.Module,
    target_modules: Union[str, List[str]],
    exclude_modules: Optional[List[str]] = None,
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    bias: str = "none",
    strict_target_match: bool = True,
) -> nn.Module:
    """
    Apply LoRA adapters to a PyTorch module.

    This function automatically freezes all parameters of the module and then
    applies LoRA adapters to the specified target layers. Supports both single
    module and multi-module scenarios (e.g., XCLIP's vision_model and mit).

    Args:
        module (nn.Module): PyTorch module to apply LoRA to.
        target_modules (Union[str, List[str]]): Names of target layers, supports
            string or list. For example: ["q_proj", "k_proj", "v_proj", "out_proj"].
        exclude_modules (Optional[List[str]], optional): List of module names to
            exclude. For example: ["encoder", "encoder_temporal"]. Defaults to None.
        r (int, optional): LoRA rank. Defaults to 8.
        lora_alpha (int, optional): LoRA alpha parameter. Defaults to 16.
        lora_dropout (float, optional): Dropout rate for LoRA layers. Defaults to 0.1.
        bias (str, optional): Bias handling strategy. Defaults to "none".
        strict_target_match (bool, optional): If True, raise RuntimeError when no
            module name matches target_modules after exclude filtering. Defaults to True.

    Returns:
        nn.Module: Module with LoRA adapters applied.

    Raises:
        ImportError: If peft library is not installed.

    Examples:
        >>> # Apply LoRA to a single module
        >>> model = apply_lora(model, target_modules=["query", "key", "value"])

        >>> # Apply LoRA to multiple modules separately (e.g., XCLIP)
        >>> vision_model = apply_lora(vision_model, ["q_proj", "k_proj"])
        >>> mit = apply_lora(mit, ["q_proj", "k_proj"])

        >>> # Use exclude_modules to exclude specific modules
        >>> model = apply_lora(
        ...     model,
        ...     target_modules=["to_q", "to_k", "to_v"],
        ...     exclude_modules=["encoder"]
        ... )
    """
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError:
        raise ImportError(
            "PEFT library is required for LoRA fine-tuning. "
            "Please install it with: pip install peft"
        )

    # Freeze all parameters (prerequisite for LoRA)
    for param in module.parameters():
        param.requires_grad = False

    # Ensure target_modules is a list
    if isinstance(target_modules, str):
        target_modules = [target_modules]

    # Discover target-module matches before injection to avoid silent mismatches.
    exclude_set = set(exclude_modules or [])
    matched_module_names: List[str] = []
    for module_name, _ in module.named_modules():
        if not module_name:
            continue
        leaf_name = module_name.rsplit(".", 1)[-1]
        if leaf_name not in target_modules:
            continue
        if any(
            module_name == ex or module_name.startswith(f"{ex}.")
            for ex in exclude_set
        ):
            continue
        matched_module_names.append(module_name)
    matched_module_names = sorted(set(matched_module_names))
    if strict_target_match and not matched_module_names:
        raise RuntimeError(
            "No module names matched LoRA target_modules after exclude filtering. "
            f"target_modules={target_modules}, exclude_modules={exclude_modules}"
        )

    # Build LoRA configuration
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=bias,
    )

    # Add exclude_modules to config if specified
    if exclude_modules:
        lora_config.exclude_modules = exclude_modules

    # Apply LoRA.
    lora_module = get_peft_model(module, lora_config)

    # Log matched targets and actual LoRA-injected module roots for auditability.
    injected_roots: List[str] = []
    for name, _ in lora_module.named_parameters():
        if "lora_A." in name:
            injected_roots.append(name.split(".lora_A.")[0])
    injected_roots = sorted(set(injected_roots))
    logging.info(
        "LoRA apply summary: target_modules=%s, matched=%d, injected=%d",
        target_modules,
        len(matched_module_names),
        len(injected_roots),
    )
    if matched_module_names:
        logging.info("LoRA matched module names: %s", matched_module_names)
    if injected_roots:
        logging.info("LoRA injected module names: %s", injected_roots)

    return lora_module
