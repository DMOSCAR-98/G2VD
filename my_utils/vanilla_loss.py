"""
Vanilla Loss (Softplus) for Binary Classification

Implements vanilla loss (softplus) for binary classification tasks.
"""
import torch
import torch.nn.functional as F


def vanilla_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Compute vanilla loss (softplus) for binary classification.

    Vanilla loss: log(1 + exp(-y * logits)) where y ∈ {-1, 1}
    For binary classification with targets in {0, 1} (same convention as BCE):
    - class 0 (label=0): loss = log(1 + exp(logits))
    - class 1 (label=1): loss = log(1 + exp(-logits))

    Vanilla loss is a smooth version of binary cross-entropy loss, providing
    smoother gradients during training. It is mathematically equivalent to BCE
    but uses softplus instead of sigmoid, which can improve numerical stability.

    Args:
        logits (torch.Tensor): Model logits, shape [B, 1] or [B].
        targets (torch.Tensor): Ground truth, shape [B], float in {0, 1}; same semantics as
            binary_cross_entropy_with_logits targets.

    Returns:
        torch.Tensor: Vanilla loss (scalar).
    """
    # Extract logits: [B, 1] -> [B]
    if logits.dim() > 1:
        logits = logits[:, 0]

    # Map {0, 1} targets to {-1, 1}
    y = 2 * targets - 1  # 0 -> -1, 1 -> +1

    # Vanilla loss: log(1 + exp(-y * logits))
    loss = torch.mean(F.softplus(-y * logits))

    return loss
