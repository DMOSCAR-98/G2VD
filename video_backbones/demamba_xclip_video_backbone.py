"""
DeMamba-XCLIP video backbone (project adaptation).

Follows the XCLIP_DeMamba path in DeMamba.py:
vision pooler_output builds global features, vision patch tokens are reordered and
processed by Mamba, then branch-wise LayerNorm is applied to global and mamba
features before concatenation and projection.

Input pixel_values: BTCHW [B, T, C, H, W]. See forward() for resolution constraints.
Intended for official Hugging Face checkpoints with default model config (same as
typical from_pretrained usage for XCLIPVisionModel).

Reference: https://github.com/chenhaoxing/DeMamba/blob/main/models/DeMamba.py
"""
from typing import Dict

import torch
import torch.nn as nn
from transformers import XCLIPVisionModel

from .mamba_base import MambaConfig, ResidualBlock


def _create_reorder_index(N, device):
    """Create reorder index for Mamba's parallel scan operation (internal helper)."""
    new_order = []
    for col in range(N):
        if col % 2 == 0:
            new_order.extend(range(col, N * N, N))
        else:
            new_order.extend(range(col + N * (N - 1), col - 1, -N))
    return torch.tensor(new_order, device=device)


def _reorder_data(data, N):
    """Reorder data helper function (internal helper)."""
    assert isinstance(data, torch.Tensor), "data should be a torch.Tensor"
    device = data.device
    new_order = _create_reorder_index(N, device)
    B, t, _, _ = data.shape
    index = new_order.repeat(B, t, 1).unsqueeze(-1)
    reordered_data = torch.gather(data, 2, index.expand_as(data))
    return reordered_data


class DeMambaXCLIPVideoBackbone(nn.Module):
    """
    Video backbone based on DeMamba-XCLIP model.

    DeMamba-XCLIP model characteristics:
    - Uses XCLIPVisionModel as vision_encoder and Mamba blocks for temporal modeling
      on reordered fused patch tokens.
    - Provides video-level representations by applying branch-wise LayerNorm to
      global features (mean of per-frame pooler_output) and mamba-processed patch
      features, then concatenating and projecting (feature_projection) to embed_dim.
    - Also returns frame_level_reps: per-frame vectors from vision_encoder pooler_output
      (optional downstream use; aligns with HF XCLIPVisionModel per-frame layout).
    - Difference from DeMamba.py XCLIP global feature definition: this adaptation uses
      mean over per-frame pooler_output, while upstream applies fc_norm2 on global
      features before concatenation with the mamba branch.
    - patch_level_reps uses vision_encoder last_hidden_state patch tokens (CLS stripped),
      same patch sequence as HF outputs before DeMamba reorder and Mamba.
    - Project adaptation vs DeMamba.py XCLIP branch: this module keeps branch-wise
      normalization semantics by normalizing global and mamba features separately
      before concatenation and final projection.
    - Patch geometry (patch_grid, patch_count, fused_patch_count) is derived from
      vision_cfg image_size and patch_size at init. fusing_ratios is fixed at 1 for
      the DeMamba XCLIP branch and must match the default hf_repo
      (xclip-base-patch16). For ViT patch16 @224: patch_count=196, fused_patch_count=196,
      concatenated dim (fused_patch_count+1)*hidden_size before projection.

    Fine-tuning modes:
    - "frozen": Freeze all parameters (default)
    - "full": Fully fine-tune all parameters
    - "mamba_only": Freeze vision_encoder; train mamba, branch LayerNorms, and feature_projection
    """

    def __init__(
        self,
        hf_repo: str = "./pretrained_weights/models--microsoft--xclip-base-patch16",
        finetune_mode: str = "frozen",
    ) -> None:
        """
        Initialize the DeMamba-XCLIP video backbone.

        Args:
            hf_repo (str): Pretrained weights source for the encoder.
                Note: this project uses from_pretrained(..., local_files_only=True)
                so it must be a local path (or an already-cached HuggingFace
                identifier). It will NOT auto-download.
            finetune_mode (str): Finetuning mode, options: "frozen", "full", or
                "mamba_only".
        """
        super().__init__()

        self.vision_encoder = XCLIPVisionModel.from_pretrained(
            hf_repo, local_files_only=True
        )

        # Config-driven metadata (aligned with HF checkpoint config).
        self.vision_cfg = self.vision_encoder.config
        self.vision_hidden_size = self.vision_cfg.hidden_size

        # DeMamba XCLIP branch: fusing_ratios=1 (see class docstring).
        self.fusing_ratios = 1
        image_size = self.vision_cfg.image_size
        patch_size = self.vision_cfg.patch_size
        if image_size % patch_size != 0:
            raise ValueError(
                f"image_size ({image_size}) must be divisible by patch_size "
                f"({patch_size})."
            )
        self.patch_grid = image_size // patch_size
        self.patch_count = self.patch_grid * self.patch_grid
        if self.patch_grid % self.fusing_ratios != 0:
            raise ValueError(
                f"patch_grid ({self.patch_grid}) must be divisible by fusing_ratios "
                f"({self.fusing_ratios})."
            )
        self.fused_grid = self.patch_grid // self.fusing_ratios
        self.fused_patch_count = self.fused_grid * self.fused_grid
        self.mamba = ResidualBlock(config=MambaConfig(d_model=self.vision_hidden_size))

        # Branch-wise normalization before feature concatenation.
        self.global_norm = nn.LayerNorm(self.vision_hidden_size)
        self.mamba_norm = nn.LayerNorm(
            self.fused_patch_count * self.vision_hidden_size
        )

        self.embed_dim = 512
        self.feature_projection = nn.Linear(
            (self.fused_patch_count + 1) * self.vision_hidden_size, self.embed_dim
        )

        self._set_finetune_mode(finetune_mode)

    def _set_finetune_mode(self, finetune_mode: str) -> None:
        """
        Set the finetuning mode for the model.

        Args:
            finetune_mode (str): Finetuning mode, options: "frozen", "full", or
                "mamba_only".
                - "frozen": Freeze all parameters.
                - "full": Fully fine-tune all parameters.
                - "mamba_only": Freeze vision_encoder; train mamba, branch LayerNorms, and
                  feature_projection.
        """
        if finetune_mode == "frozen":
            for param in self.parameters():
                param.requires_grad = False
        elif finetune_mode == "full":
            for param in self.parameters():
                param.requires_grad = True
        elif finetune_mode == "mamba_only":
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
            for param in self.mamba.parameters():
                param.requires_grad = True
            for param in self.global_norm.parameters():
                param.requires_grad = True
            for param in self.mamba_norm.parameters():
                param.requires_grad = True
            for param in self.feature_projection.parameters():
                param.requires_grad = True
        else:
            raise ValueError(
                f"Unsupported finetune mode: {finetune_mode}. "
                f"Please choose from: ['frozen', 'full', 'mamba_only']"
            )

    def forward(
        self,
        pixel_values: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through XCLIP vision encoder and DeMamba temporal pooling.

        Args:
            pixel_values (torch.Tensor): Video input, shape [B, T, C, H, W] (BTCHW).
                Default pretrained checkpoint expects:
                - T = num_frames (flexible; no fixed temporal position encoding).
                - C = num_channels (typically 3 for RGB).
                - H, W = image_size (typically 224x224).
                Input shape validation is delegated to the underlying HF model.
                Vision outputs use attribute access (pooler_output, last_hidden_state)
                as with default official checkpoints.

        Returns:
            Dict[str, torch.Tensor]: Video features.
                - "video_level_reps": Branch-wise LayerNorm on global and mamba features,
                  followed by concatenation and projection to embed_dim, shape
                  [B, embed_dim] (default embed_dim=512; concatenated feature dim is
                  151296 before projection).
                - "frame_level_reps": per-frame vectors from vision_encoder
                  pooler_output, shape [B, T, vision_hidden_size].
                - "patch_level_reps": per-patch vectors from vision_encoder
                  last_hidden_state (excluding CLS), shape [B, T, patch_count,
                  vision_hidden_size] when input matches vision_cfg.image_size.
        """
        batch_size, num_frames, num_channels, height, width = pixel_values.shape
        pixel_values = pixel_values.reshape(-1, num_channels, height, width)

        # Get encoder outputs.
        vision_outputs = self.vision_encoder(pixel_values=pixel_values)
        frame_tokens = vision_outputs.pooler_output
        frame_level_reps = frame_tokens.reshape(batch_size, num_frames, -1)
        patch_tokens = vision_outputs.last_hidden_state[:, 1:, :]
        patch_level_reps = patch_tokens.reshape(
            batch_size, num_frames, self.patch_count, self.vision_hidden_size
        )
        global_features = frame_level_reps.mean(1)
        patch_tokens = patch_tokens.reshape(
            batch_size,
            num_frames,
            self.fusing_ratios,
            self.fused_grid,
            self.fusing_ratios,
            self.fused_grid,
            self.vision_hidden_size,
        )
        mamba_inputs = patch_tokens.permute(0, 2, 4, 1, 3, 5, 6).contiguous().reshape(
            batch_size * self.fused_patch_count,
            num_frames,
            -1,
            self.vision_hidden_size,
        )
        mamba_batch = batch_size * self.fused_patch_count

        # Reorder and process through mamba.
        mamba_inputs = _reorder_data(mamba_inputs, self.fusing_ratios)
        mamba_inputs = (
            mamba_inputs.permute(0, 2, 1, 3).contiguous().reshape(
                mamba_batch, -1, self.vision_hidden_size
            )
        )
        mamba_outputs = self.mamba(mamba_inputs)

        # Aggregate mamba features.
        mamba_features = mamba_outputs.mean(1)
        mamba_features = mamba_features.reshape(batch_size, -1)

        # Normalize each branch independently, then concatenate and project.
        global_features = self.global_norm(global_features)
        mamba_features = self.mamba_norm(mamba_features)
        concatenated_features = torch.cat((global_features, mamba_features), dim=1)
        video_level_reps = self.feature_projection(concatenated_features)

        return {
            "video_level_reps": video_level_reps,
            "frame_level_reps": frame_level_reps,
            "patch_level_reps": patch_level_reps,
        }
