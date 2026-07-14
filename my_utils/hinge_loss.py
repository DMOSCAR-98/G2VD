"""
Hinge Loss for Binary Classification

Implements hinge loss for binary classification tasks with configurable margin.
"""
import torch


def hinge_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """
    Compute hinge loss for binary classification with configurable margin.

    Hinge loss: max(0, margin - y * logits) where y ∈ {-1, 1}
    For binary classification with targets in {0, 1} (same convention as BCE):
    - class 0 (label=0): loss = max(0, margin + logits)
      When logits < -margin, loss = 0 (sufficiently confident as class 0)
      When logits >= -margin, loss = margin + logits (penalize uncertainty)
    - class 1 (label=1): loss = max(0, margin - logits)
      When logits > margin, loss = 0 (sufficiently confident as class 1)
      When logits <= margin, loss = margin - logits (penalize uncertainty)

    The margin parameter defines a "safety threshold" that controls the minimum
    confidence required for correct classification. Only samples with |logits| < margin
    are penalized. Larger margin requires higher confidence to avoid penalty, which
    can improve generalization by encouraging a more robust decision boundary.
    However, margin that is too large relative to logits range may cause underfitting,
    as the model struggles to satisfy the high confidence requirement for all samples.
    Margin that is too small may cause overfitting by allowing samples too close to
    the decision boundary.

    Args:
        logits (torch.Tensor): Model logits, shape [B, 1] or [B].
        targets (torch.Tensor): Ground truth, shape [B], float in {0, 1}; same semantics as
            binary_cross_entropy_with_logits targets.
        margin (float): Safety margin for hinge loss. Defines the threshold below which
            samples are penalized. Larger margin requires higher confidence (larger
            |logits|) to avoid penalty, which can improve generalization but may cause
            underfitting if too large. Smaller margin may cause overfitting by allowing
            samples too close to the decision boundary. Should be proportional to the
            typical range of logits to balance between preventing overfitting and
            avoiding underfitting. Common values range from 0.5 to 5.0, depending on
            the logits scale. Defaults to 1.0.

    Returns:
        torch.Tensor: Hinge loss (scalar).

    Examples:
        >>> # Standard usage with default margin
        >>> loss = hinge_loss(logits, targets)

        >>> # With custom margin (adjust based on your logits range)
        >>> loss = hinge_loss(logits, targets, margin=2.0)
    """
    # Extract logits: [B, 1] -> [B]
    if logits.dim() > 1:
        logits = logits[:, 0]

    # Map {0, 1} targets to hinge labels {-1, 1}
    y = 2 * targets - 1  # 0 -> -1, 1 -> +1

    # Hinge loss: max(0, margin - y * logits)
    loss = torch.mean(torch.relu(margin - y * logits))

    return loss
