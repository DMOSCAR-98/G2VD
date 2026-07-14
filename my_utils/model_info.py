"""
Logging helpers for model structure: Linear layer names and trainable parameter counts.
"""
import logging

import torch
import torch.nn as nn

from typing import Dict, List, Tuple, Union


def linear_layer_info(model: nn.Module) -> List[str]:
    """
    Output names of all Linear layers in the model via logging.

    Iterate through all modules in the model, find all nn.Linear layers,
    and output their names via logging. Returns a list of all linear
    layer names.

    Args:
        model (nn.Module): PyTorch model object.

    Returns:
        List[str]: List of all Linear layer names, sorted alphabetically.

    Examples:
        >>> import torch.nn as nn
        >>> model = nn.Sequential(
        ...     nn.Linear(10, 20),
        ...     nn.ReLU(),
        ...     nn.Linear(20, 1)
        ... )
        >>> linear_layers = linear_layer_info(model)
        >>> # Output: ['0', '2']
    """

    linear_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linear_layers.append(name)

    # Output via logging
    logging.info(f"\n{'=' * 60}")
    logging.info(f"Found {len(linear_layers)} Linear layers in model:")
    logging.info(f"{'=' * 60}")
    for name in sorted(linear_layers):
        logging.info(f"  - {name}")
    logging.info(f"{'=' * 60}\n")

    return linear_layers


def trainable_parameter_info(
    model: nn.Module,
    show_layer_info: bool = True,
) -> Dict[str, Union[int, float, List[Tuple[str, int, torch.Size]]]]:
    """
    Output trainable layers and parameter statistics via logging.

    Count the number of trainable and frozen parameters in the model,
    calculate the percentage of trainable parameters. Similar to peft's
    print_trainable_parameters(), but with additional layer information
    output functionality. Can display detailed information (name, parameter
    count, shape) for each trainable and frozen layer.

    Args:
        model (nn.Module): PyTorch model object.
        show_layer_info (bool, optional): Whether to show detailed information
            for each trainable layer. If True, outputs name, parameter count,
            and shape for each trainable and frozen layer. Defaults to True.

    Returns:
        Dict[str, Union[int, float, List[Tuple[str, int, torch.Size]]]]:
            Dictionary containing the following keys:
            - 'trainable_params' (int): Total number of trainable parameters.
            - 'all_params' (int): Total number of all parameters.
            - 'trainable_percentage' (float): Share of trainable params.
            - 'trainable_layers' (List[Tuple[str, int, torch.Size]]):
                List of trainable layers, each element is (layer_name,
                param_count, param_shape).
            - 'frozen_layers' (List[Tuple[str, int, torch.Size]]):
                List of frozen layers, each element is (layer_name,
                param_count, param_shape).

    Examples:
        >>> import torch.nn as nn
        >>> model = nn.Sequential(
        ...     nn.Linear(10, 20),
        ...     nn.Linear(20, 1)
        ... )
        >>> # Freeze the first linear layer
        >>> for param in model[0].parameters():
        ...     param.requires_grad = False
        >>> stats = trainable_parameter_info(model, show_layer_info=True)
        >>> print(f"Trainable: {stats['trainable_params']:,}")
        >>> print(f"Total: {stats['all_params']:,}")
        >>> print(f"Percentage: {stats['trainable_percentage']:.2f}%")
    """

    trainable_params = 0
    all_params = 0
    trainable_layers = []
    frozen_layers = []

    # Count parameters and collect layer information
    for name, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            trainable_layers.append((name, param.numel(), param.shape))
        else:
            frozen_layers.append((name, param.numel(), param.shape))

    # Calculate percentage
    trainable_percentage = (
        100 * trainable_params / all_params if all_params > 0 else 0
    )

    # Output statistics via logging
    logging.info(f"\n{'=' * 60}")
    logging.info("Trainable Parameters Statistics")
    logging.info(f"{'=' * 60}")
    logging.info(
        f"Trainable params: {trainable_params:,} || "
        f"All params: {all_params:,} || "
        f"Trainable%: {trainable_percentage:.4f}%"
    )
    logging.info(f"{'=' * 60}")

    # Output trainable layer information
    if show_layer_info:
        logging.info(f"\nTrainable Layers ({len(trainable_layers)} layers):")
        logging.info(f"{'-' * 60}")
        for name, num_params, shape in sorted(trainable_layers):
            logging.info(
                f"  {name:60s} | Params: {num_params:>12,} | "
                f"Shape: {str(shape):>30s}"
            )

        logging.info(f"\nFrozen Layers ({len(frozen_layers)} layers):")
        logging.info(f"{'-' * 60}")
        for name, num_params, shape in sorted(frozen_layers):
            logging.info(
                f"  {name:60s} | Params: {num_params:>12,} | "
                f"Shape: {str(shape):>30s}"
            )

    logging.info(f"{'=' * 60}\n")

    return {
        "trainable_params": trainable_params,
        "all_params": all_params,
        "trainable_percentage": trainable_percentage,
        "trainable_layers": trainable_layers,
        "frozen_layers": frozen_layers
    }
