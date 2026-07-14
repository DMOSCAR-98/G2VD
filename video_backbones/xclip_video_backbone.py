"""
X-CLIP video backbone (Hugging Face transformers).

Follows the vision + MIT path in modeling_x_clip.py (XCLIPModel.get_video_features):
per-frame pooler_output passes through visual_projection, then MIT pools over time.

Input pixel_values: BTCHW [B, T, C, H, W]. See forward() for required T and resolution.
Intended for official Hugging Face checkpoints with default model config (same as
typical from_pretrained usage in modeling_x_clip.py).

Reference: https://github.com/huggingface/transformers/blob/main/src/transformers/models/x_clip/modeling_x_clip.py
"""
from typing import Dict

import torch
import torch.nn as nn
from transformers import AutoModel


class XCLIPVideoBackbone(nn.Module):
    """
    Video backbone based on X-CLIP model.

    X-CLIP model characteristics:
    - Provides video-level representations (MIT pooled output; aligns with
      XCLIPModel.get_video_features in Hugging Face transformers).
    - Also returns frame_level_reps: per-frame vectors after visual_projection and
      before MIT (for optional downstream use).
    - patch_level_reps uses vision last_hidden_state patch tokens (CLS stripped),
      same source as in HF XCLIPVisionModel outputs, not visual_projection.

    Fine-tuning modes:
    - "frozen": Freeze all parameters (default)
    - "lora": Apply LoRA fine-tuning to attention linear layers in vision_model
      and mit (q_proj, k_proj, v_proj, out_proj); visual_projection stays frozen
    - "full": Fully fine-tune vision_model, visual_projection, and mit
    - "mit_only": Freeze vision_model and visual_projection; train mit only
    """

    def __init__(
        self,
        hf_repo: str = "./pretrained_weights/models--microsoft--xclip-base-patch16",
        finetune_mode: str = "frozen",
    ) -> None:
        """
        Initialize the X-CLIP video backbone.

        Args:
            hf_repo (str): Pretrained weights source for the backbone.
                Note: this project uses from_pretrained(..., local_files_only=True)
                so it must be a local path (or an already-cached HuggingFace
                identifier). It will NOT auto-download.
                Loading uses a full XCLIPModel checkpoint; only vision_model,
                visual_projection, and mit are kept on this module.
            finetune_mode (str): Finetuning mode, options: "frozen", "lora", "full",
                or "mit_only".
        """
        super().__init__()

        full_model = AutoModel.from_pretrained(hf_repo, local_files_only=True)
        self.vision_model, self.visual_projection, self.mit = (
            full_model.vision_model,
            full_model.visual_projection,
            full_model.mit,
        )

        # Config-driven metadata (aligned with HF checkpoint config).
        self.vision_cfg = full_model.config.vision_config
        self.projection_dim = full_model.config.projection_dim
        self.embed_dim = self.vision_cfg.mit_hidden_size
        self.vision_hidden_size = self.vision_cfg.hidden_size

        del full_model

        self._set_finetune_mode(finetune_mode)

    def _set_finetune_mode(self, finetune_mode: str) -> None:
        """
        Set the finetuning mode for the model.

        Args:
            finetune_mode (str): Finetuning mode, options: "frozen", "lora", "full",
                or "mit_only".
                - "frozen": Freeze all parameters.
                - "lora": Apply LoRA fine-tuning to attention linear layers in
                  vision_model and mit; visual_projection stays frozen.
                - "full": Fully fine-tune all parameters.
                - "mit_only": Only mit is trainable; vision_model and
                  visual_projection stay frozen.
        """
        if finetune_mode == "frozen":
            for param in self.vision_model.parameters():
                param.requires_grad = False
            for param in self.visual_projection.parameters():
                param.requires_grad = False
            for param in self.mit.parameters():
                param.requires_grad = False
        elif finetune_mode == "lora":
            from my_utils import apply_lora

            self.vision_model = apply_lora(
                self.vision_model, ["q_proj", "k_proj", "v_proj", "out_proj"]
            )
            for param in self.visual_projection.parameters():
                param.requires_grad = False
            self.mit = apply_lora(
                self.mit, ["q_proj", "k_proj", "v_proj", "out_proj"]
            )
        elif finetune_mode == "full":
            for param in self.vision_model.parameters():
                param.requires_grad = True
            for param in self.visual_projection.parameters():
                param.requires_grad = True
            for param in self.mit.parameters():
                param.requires_grad = True
        elif finetune_mode == "mit_only":
            for param in self.vision_model.parameters():
                param.requires_grad = False
            for param in self.visual_projection.parameters():
                param.requires_grad = False
            for param in self.mit.parameters():
                param.requires_grad = True
        else:
            raise ValueError(
                f"Unsupported finetune mode: {finetune_mode}. "
                f"Please choose from: ['frozen', 'lora', 'full', 'mit_only']"
            )

    def forward(
        self,
        pixel_values: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the X-CLIP vision_model, visual_projection, and mit.

        Args:
            pixel_values (torch.Tensor): Video input, shape [B, T, C, H, W] (BTCHW).
                Default pretrained checkpoint expects:
                - T = num_frames (typically 8 for base X-CLIP weights).
                - C = num_channels (typically 3 for RGB).
                - H, W = image_size (typically 224x224).
                Input shape validation is delegated to the underlying HF model.
                Vision outputs use attribute access (pooler_output, last_hidden_state)
                as with default official checkpoints.

        Returns:
            Dict[str, torch.Tensor]: Video features.
                - "video_level_reps": MIT pooled video-level representation,
                  shape [B, embed_dim].
                - "frame_level_reps": per-frame vectors after visual_projection (MIT input),
                  shape [B, T, projection_dim].
                - "patch_level_reps": per-patch vectors from vision_model
                  last_hidden_state (excluding CLS), shape [B, T, Np,
                  vision_hidden_size]; matches HF XCLIPVisionModel patch sequence
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

        mit_outputs = self.mit(frame_level_reps)
        video_level_reps = mit_outputs.pooler_output

        return {
            "video_level_reps": video_level_reps,
            "frame_level_reps": frame_level_reps,
            "patch_level_reps": patch_level_reps,
        }
