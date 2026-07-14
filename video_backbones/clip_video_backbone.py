"""
CLIP video backbone (Hugging Face transformers).

Follows the vision path in modeling_clip.py (CLIPModel.get_image_features):
per-frame pooler_output passes through visual_projection, then frame features
are mean-pooled over time.

Input pixel_values: BTCHW [B, T, C, H, W]. See forward() for required resolution.
Intended for official Hugging Face checkpoints with default model config (same as
typical from_pretrained usage in modeling_clip.py).

Reference: https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip/modeling_clip.py
"""
from typing import Dict

import torch
import torch.nn as nn
from transformers import AutoModel


class CLIPVideoBackbone(nn.Module):
    """
    Video backbone based on CLIP model.

    CLIP model characteristics:
    - Provides video-level representations by mean pooling per-frame projected
      CLIP vision features over time.
    - Also returns frame_level_reps: per-frame vectors after visual_projection
      (for optional downstream use).
    - patch_level_reps uses vision_model last_hidden_state patch tokens (CLS stripped),
      same source as in HF CLIPVisionModel outputs, not visual_projection.

    Fine-tuning modes:
    - "frozen": Freeze all parameters (default)
    - "full": Fully fine-tune vision_model and visual_projection
    """

    def __init__(
        self,
        hf_repo: str = "./pretrained_weights/models--openai--clip-vit-base-patch16",
        finetune_mode: str = "frozen",
    ) -> None:
        """
        Initialize the CLIP video backbone.

        Args:
            hf_repo (str): Pretrained weights source for the backbone.
                Note: this project uses from_pretrained(..., local_files_only=True)
                so it must be a local path (or an already-cached HuggingFace
                identifier). It will NOT auto-download.
                Loading uses a full CLIPModel checkpoint; only vision_model and
                visual_projection are kept on this module.
            finetune_mode (str): Finetuning mode, options: "frozen" or "full".
        """
        super().__init__()

        full_model = AutoModel.from_pretrained(hf_repo, local_files_only=True)
        self.vision_model, self.visual_projection = (
            full_model.vision_model,
            full_model.visual_projection,
        )

        # Config-driven metadata (aligned with HF checkpoint config).
        self.vision_cfg = full_model.config.vision_config
        self.projection_dim = full_model.config.projection_dim
        self.embed_dim = self.projection_dim
        self.vision_hidden_size = self.vision_cfg.hidden_size

        del full_model

        self._set_finetune_mode(finetune_mode)

    def _set_finetune_mode(self, finetune_mode: str) -> None:
        """
        Set the finetuning mode for the model.

        Args:
            finetune_mode (str): Finetuning mode, options: "frozen" or "full".
                - "frozen": Freeze all parameters.
                - "full": Fully fine-tune all parameters.
        """
        if finetune_mode == "frozen":
            for param in self.vision_model.parameters():
                param.requires_grad = False
            for param in self.visual_projection.parameters():
                param.requires_grad = False
        elif finetune_mode == "full":
            for param in self.vision_model.parameters():
                param.requires_grad = True
            for param in self.visual_projection.parameters():
                param.requires_grad = True
        else:
            raise ValueError(
                f"Unsupported finetune mode: {finetune_mode}. "
                f"Please choose from: ['frozen', 'full']"
            )

    def forward(
        self,
        pixel_values: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through CLIP vision_model and temporal mean pooling.

        Args:
            pixel_values (torch.Tensor): Video input, shape [B, T, C, H, W] (BTCHW).
                Default pretrained checkpoint expects:
                - T = num_frames (flexible; CLIP ViT is applied per frame).
                - C = num_channels (typically 3 for RGB).
                - H, W = image_size (typically 224x224).
                Input shape validation is delegated to the underlying HF model.
                Vision outputs use attribute access (pooler_output, last_hidden_state)
                as with default official checkpoints.

        Returns:
            Dict[str, torch.Tensor]: Video features.
                - "video_level_reps": mean-pooled per-frame representation,
                  shape [B, embed_dim].
                - "frame_level_reps": per-frame vectors after visual_projection,
                  shape [B, T, projection_dim].
                - "patch_level_reps": per-patch vectors from vision_model
                  last_hidden_state (excluding CLS), shape [B, T, Np,
                  vision_hidden_size]; matches HF CLIPVisionModel patch sequence
                  before visual_projection (Np = num_patches per frame).
        """
        batch_size, num_frames, num_channels, height, width = pixel_values.shape
        pixel_values = pixel_values.reshape(-1, num_channels, height, width)

        vision_outputs = self.vision_model(pixel_values=pixel_values)
        frame_tokens = self.visual_projection(vision_outputs.pooler_output)
        frame_level_reps = frame_tokens.reshape(batch_size, num_frames, -1)
        patch_tokens = vision_outputs.last_hidden_state[:, 1:, :]
        patch_level_reps = patch_tokens.reshape(
            batch_size, num_frames, -1, self.vision_hidden_size
        )
        video_level_reps = frame_level_reps.mean(1)

        return {
            "video_level_reps": video_level_reps,
            "frame_level_reps": frame_level_reps,
            "patch_level_reps": patch_level_reps,
        }
