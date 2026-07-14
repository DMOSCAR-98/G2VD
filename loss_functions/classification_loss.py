"""
Classification Loss for Causal Disentanglement

Provides binary classification loss for AI-generated video detection in the Causal
Disentanglement (CD) setup. Used to train heads on logits produced from
causal and non-causal branches (for example "causal_logits" / "non_causal_logits" from
CausalDisentanglementClassifier).

For AI-generated video detection: label 0 = real, label 1 = fake (AI-generated).
Default loss_type is BCE. The focal branch uses fixed alpha and gamma literals in
the sigmoid_focal_loss call; change those literals in code when tuning focal loss.
"""
import torch
import torch.nn.functional as F
import torchvision.ops

from my_utils import hinge_loss, vanilla_loss


def classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_type: str = "bce",
) -> torch.Tensor:
    """
    Binary classification loss for logits vs labels.

    Used for CD classification terms on logits from either branch. BCE with logits
    uses per-sample scores s (entries of logits) with
    -E[Y log(sigmoid(s)) + (1-Y) log(1-sigmoid(s))].
    Supports BCE, Hinge, Vanilla, and Focal loss for AI-generated video detection.

    Args:
        logits (torch.Tensor): Binary logits, shape [B, 1] or [B].
        labels (torch.Tensor): Ground truth labels, shape [B]; 0 = real, 1 = fake (AI-generated).
            Converted to float targets once; all loss_type values use the same targets
            as BCE (including hinge and vanilla).
        loss_type (str): "bce", "hinge", "vanilla", "focal". Defaults to "bce".

    Returns:
        torch.Tensor: Scalar classification loss.
    """
    if logits.dim() > 1 and logits.shape[-1] != 1:
        raise ValueError(
            f"classification_loss expects binary logits with last dim 1,"
            f"got shape {tuple(logits.shape)}."
        )

    logits_binary = logits if logits.dim() == 1 else logits.squeeze(-1)
    targets = labels.float()

    if loss_type == "bce":
        return F.binary_cross_entropy_with_logits(
            logits_binary, targets
        )
    elif loss_type == "hinge":
        return hinge_loss(logits_binary, targets)
    elif loss_type == "vanilla":
        return vanilla_loss(logits_binary, targets)
    elif loss_type == "focal":
        # alpha=1/5, gamma=1.0: edit these literals here when tuning focal loss.
        focal_loss = torchvision.ops.sigmoid_focal_loss(
            inputs=logits_binary,
            targets=targets,
            alpha=1.0 / 5.0,
            gamma=1.0,
            reduction="mean",
        )
        return focal_loss
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")
