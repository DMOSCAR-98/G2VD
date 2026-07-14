"""
TimeSformer baseline model with unified BTCHW -> logits interface.
"""
import torch
import torch.nn as nn
from transformers import TimesformerModel


class TimeSformer(nn.Module):
    """
    TimeSformer baseline for binary video classification.

    Input layout is BTCHW [B, T, C, H, W]. The model uses
    last_hidden_state[:, 0, :] (CLS token) as video representation, then a
    linear classifier to produce logits.
    """

    def __init__(
        self,
        hf_repo: str = "./pretrained_weights/models--facebook--timesformer-base-finetuned-k400",
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        self.backbone = TimesformerModel.from_pretrained(
            hf_repo, local_files_only=True
        )
        hidden_size: int = self.backbone.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            pixel_values (torch.Tensor): Video input [B, T, C, H, W] (BTCHW).
                Default pretrained checkpoint expects:
                - T = num_frames (typically 8 for base TimeSformer weights).
                - C = num_channels (typically 3 for RGB).
                - H, W = image_size (typically 224x224).
                Input shape validation is delegated to the underlying HF model.

        Returns:
            torch.Tensor: Logits, shape [B, num_classes].
        """
        outputs = self.backbone(pixel_values=pixel_values)
        logits = self.classifier(outputs.last_hidden_state[:, 0, :])
        return logits
