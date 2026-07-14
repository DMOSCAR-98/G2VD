"""
Training losses for detection and causal disentanglement.

- classification_loss: binary classification (BCE, hinge, vanilla, focal) on logits.
- independence_loss: HSIC (L_ind) between causal and non-causal feature branches.

Prefer: from loss_functions import classification_loss, independence_loss.
"""

from .classification_loss import classification_loss
from .independence_loss import independence_loss

__all__ = [
    "classification_loss",
    "independence_loss",
]
