"""
Scalar scheduling over training steps: linear, step, or sigmoid curves.

Used for dynamic loss weights, learning-rate multipliers, and similar hyperparameters.
"""
import math


def dynamic_parameter_scheduler(
    strategy: str,
    step: int,
    total_steps: int,
    initial_value: float,
    final_value: float,
    step_threshold: float = 0.5,
    sigmoid_midpoint: float = 0.5,
    sigmoid_scale: float = 0.1,
) -> float:
    """
    Dynamic parameter scheduler with multiple scheduling strategies.

    Supports three scheduling strategies:
    - Linear: value = initial + (final - initial) * progress,
      progress = (step-1)/(total_steps-1). Linear over steps.
    - Step: value = final_value if progress > threshold else initial_value.
      Step function that switches when progress exceeds threshold.
    - Sigmoid: value = initial + (final - initial) * sigmoid((progress - midpoint)
      / scale), sigmoid(x) = 1/(1+exp(-x)). Smooth S-shaped transition.

    Args:
        strategy (str): "linear", "step", or "sigmoid".
        step (int): Current step number (1-indexed, starting from 1).
        total_steps (int): Total number of steps.
        initial_value (float): Parameter value at the first step.
        final_value (float): Parameter value at the last step.
        step_threshold (float): Threshold ratio for step strategy (0.0 to 1.0).
            If progress > step_threshold return final_value else initial_value,
            where progress = (step - 1) / (total_steps - 1). Defaults to 0.5.
        sigmoid_midpoint (float): Sigmoid center ratio (0.0 to 1.0).
            The sigmoid curve centers around this point. Defaults to 0.5.
        sigmoid_scale (float): Sigmoid scale; smaller = steeper. Default 0.1.

    Returns:
        float: Current parameter value based on the selected strategy.

    Examples:
        >>> weight = dynamic_parameter_scheduler(
        ...     strategy="linear",
        ...     step=1,
        ...     total_steps=20,
        ...     initial_value=0.5,
        ...     final_value=1.5,
        ... )
        >>> weight
        0.5
    """
    if total_steps <= 1:
        return final_value

    # Calculate progress from 0.0 (first step) to 1.0 (last step).
    progress = (step - 1) / (total_steps - 1)

    if strategy == "linear":
        current_value = initial_value + (final_value - initial_value) * progress
    elif strategy == "step":
        current_value = final_value if progress > step_threshold else initial_value
    elif strategy == "sigmoid":
        sigmoid_input = (progress - sigmoid_midpoint) / sigmoid_scale
        sigmoid_output = 1.0 / (1.0 + math.exp(-sigmoid_input))
        current_value = initial_value + (final_value - initial_value) * sigmoid_output
    else:
        raise ValueError(
            f"Unsupported strategy: {strategy}. "
            f"Please choose from: 'linear', 'step', 'sigmoid'"
        )

    return current_value
