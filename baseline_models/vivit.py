"""
ViViT baseline model with unified BTCHW -> logits interface.
"""
import torch
import torch.nn as nn
from transformers import VivitModel


class ViViT(nn.Module):
    """
    ViViT baseline for binary video classification.

    Input layout is BTCHW [B, T, C, H, W]. The model uses
    last_hidden_state[:, 0, :] (CLS token) as video representation, then a
    linear classifier to produce logits.
    """

    def __init__(
        self,
        hf_repo: str = "./pretrained_weights/models--google--vivit-b-16x2-kinetics400",
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        self.backbone = VivitModel.from_pretrained(
            hf_repo,
            local_files_only=True,
            add_pooling_layer=False,
        )
        hidden_size: int = self.backbone.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            pixel_values (torch.Tensor): Video input [B, T, C, H, W] (BTCHW).
                Default pretrained checkpoint expects:
                - T = num_frames (typically 32 for base ViViT weights).
                - C = num_channels (typically 3 for RGB).
                - H, W = image_size (typically 224x224).
                Input shape validation is delegated to the underlying HF model.

        Returns:
            torch.Tensor: Logits, shape [B, num_classes].
        """
        outputs = self.backbone(pixel_values=pixel_values)
        logits = self.classifier(outputs.last_hidden_state[:, 0, :])
        return logits
