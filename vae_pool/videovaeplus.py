"""
VideoVAEPlus: unified VideoVAEPlus models (videovaeplus_16z, videovaeplus_4z,
videovaeplus_16zcap, videovaeplus_4zcap).

Single entry point for all VideoVAEPlus variants. Instantiate with model_type only;
checkpoint path is resolved from VIDEOVAEPLUS_CHECKPOINT_MAP (same pattern as TAEHV).
Missing checkpoint files raise at torch.load (same as TAEHV/TAESDV) for inference-only use.
Caption-guided models use T5Embedder; its Hugging Face cache root is set in the
constructor and is separate from VIDEOVAEPLUS_CHECKPOINT_MAP.
"""
import warnings
from typing import Any, Dict, List, Optional, Union

import torch

from .videovaeplus_src.models.autoencoder2plus1d_1dcnn import (
    AutoencoderKL2plus1D_1dcnn_Standard,
)

# Config mapping: model_type -> full config (ddconfig, ppconfig, embed_dim, etc.)
# Separated from checkpoint mapping for clarity.
# Model type series: videovaeplus_16z, videovaeplus_4z, videovaeplus_16zcap, videovaeplus_4zcap.
VIDEOVAEPLUS_CONFIG_MAP: Dict[str, Dict[str, Any]] = {
    "videovaeplus_16z": {
        "use_quant_conv": False,
        "embed_dim": 0,
        "caption_guide": False,
        "ddconfig": {
            "double_z": True,
            "z_channels": 16,
            "resolution": 224,
            "in_channels": 3,
            "out_ch": 3,
            "ch": 128,
            "ch_mult": [1, 2, 4, 4],
            "temporal_down_factor": 1,
            "num_res_blocks": 2,
            "attn_resolutions": [],
            "dropout": 0.0,
        },
        "ppconfig": {
            "temporal_scale_factor": 4,
            "z_channels": 16,
            "out_ch": 16,
            "ch": 16,
            "attn_temporal_factor": [],
        },
    },
    "videovaeplus_4z": {
        "use_quant_conv": True,
        "embed_dim": 4,
        "caption_guide": False,
        "ddconfig": {
            "double_z": True,
            "z_channels": 4,
            "resolution": 224,
            "in_channels": 3,
            "out_ch": 3,
            "ch": 128,
            "ch_mult": [1, 2, 4, 4],
            "temporal_down_factor": 1,
            "num_res_blocks": 2,
            "attn_resolutions": [],
            "dropout": 0.0,
        },
        "ppconfig": {
            "temporal_scale_factor": 4,
            "z_channels": 4,
            "out_ch": 4,
            "ch": 4,
            "attn_temporal_factor": [],
        },
    },
    "videovaeplus_16zcap": {
        "use_quant_conv": False,
        "embed_dim": 0,
        "caption_guide": True,
        "t5_model_max_length": 120,
        "ddconfig": {
            "double_z": True,
            "z_channels": 16,
            "resolution": 224,
            "in_channels": 3,
            "out_ch": 3,
            "ch": 128,
            "ch_mult": [1, 2, 4, 4],
            "temporal_down_factor": 1,
            "num_res_blocks": 2,
            "attn_resolutions": [28, 56, 112, 224],
            "dropout": 0.0,
        },
        "ppconfig": {
            "temporal_scale_factor": 4,
            "z_channels": 16,
            "out_ch": 16,
            "ch": 16,
            "attn_temporal_factor": [2, 4],
        },
    },
    "videovaeplus_4zcap": {
        "use_quant_conv": True,
        "embed_dim": 4,
        "caption_guide": True,
        "t5_model_max_length": 120,
        "ddconfig": {
            "double_z": True,
            "z_channels": 4,
            "resolution": 224,
            "in_channels": 3,
            "out_ch": 3,
            "ch": 128,
            "ch_mult": [1, 2, 4, 4],
            "temporal_down_factor": 1,
            "num_res_blocks": 2,
            "attn_resolutions": [28, 56, 112, 224],
            "dropout": 0.0,
        },
        "ppconfig": {
            "temporal_scale_factor": 4,
            "z_channels": 4,
            "out_ch": 4,
            "ch": 4,
            "attn_temporal_factor": [2, 4],
        },
    },
}

# Checkpoint mapping: model_type -> checkpoint path (same pattern as TAEHV).
VIDEOVAEPLUS_CHECKPOINT_MAP: Dict[str, str] = {
    "videovaeplus_16z": "./pretrained_weights/vae_pool/videovaeplus/sota-4-16z.ckpt",
    "videovaeplus_4z": "./pretrained_weights/vae_pool/videovaeplus/sota-4-4z.ckpt",
    "videovaeplus_16zcap": "./pretrained_weights/vae_pool/videovaeplus/sota-4-16z-text.ckpt",
    "videovaeplus_4zcap": "./pretrained_weights/vae_pool/videovaeplus/sota-4-4z-text.ckpt",
}


class VideoVAEPlus(AutoencoderKL2plus1D_1dcnn_Standard):
    """
    Unified VideoVAEPlus model supporting videovaeplus_16z, videovaeplus_4z,
    videovaeplus_16zcap, videovaeplus_4zcap variants.

    Instantiate with model_type only; checkpoint path is resolved from
    VIDEOVAEPLUS_CHECKPOINT_MAP. Same pattern as TAEHV: torch.load runs unconditionally;
    a missing file fails construction (inference expects pretrained weights).

    Args:
        model_type (str): One of "videovaeplus_16z", "videovaeplus_4z",
            "videovaeplus_16zcap", "videovaeplus_4zcap".
            - videovaeplus_16z, videovaeplus_4z: No caption guidance.
            - videovaeplus_16zcap, videovaeplus_4zcap: Caption-guided (T5), accept captions in forward.
    """

    def __init__(self, model_type: str = "videovaeplus_16z") -> None:
        if model_type not in VIDEOVAEPLUS_CONFIG_MAP:
            raise ValueError(
                f"Unsupported model_type: {model_type}. "
                f"Choose from: {list(VIDEOVAEPLUS_CONFIG_MAP.keys())}"
            )
        if model_type not in VIDEOVAEPLUS_CHECKPOINT_MAP:
            raise ValueError(
                f"No checkpoint mapping for model_type: {model_type}. "
                f"VIDEOVAEPLUS_CHECKPOINT_MAP must include it."
            )

        self.model_type = model_type
        cfg = VIDEOVAEPLUS_CONFIG_MAP[model_type]
        ddconfig = cfg["ddconfig"]
        ppconfig = cfg["ppconfig"]
        embed_dim = cfg.get("embed_dim", 0)
        use_quant_conv = cfg["use_quant_conv"]
        caption_guide = cfg["caption_guide"]
        t5_model_max_length = cfg.get("t5_model_max_length", 120)

        super().__init__(
            ddconfig=ddconfig,
            ppconfig=ppconfig,
            embed_dim=embed_dim,
            use_quant_conv=use_quant_conv,
            caption_guide=caption_guide,
            t5_model_max_length=t5_model_max_length,
        )

        # Initialize T5 embedder for caption-guided models
        self.text_embedder = None
        if caption_guide:
            try:
                from .videovaeplus_src.modules.t5 import T5Embedder

                # Use cuda:0 to ensure the model is instantiated on the main device.
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                # T5 lives under the project pretrained root, not under vae_pool checkpoints.
                self.text_embedder = T5Embedder(
                    device=device,
                    dir_or_name="models--google--flan-t5-large",
                    local_cache=True,
                    cache_dir="./pretrained_weights",
                    model_max_length=self.t5_model_max_length,
                )
            except ImportError:
                raise ImportError(
                    "T5Embedder is required for caption-guided models. "
                    "Please ensure vae_pool.videovaeplus_src.modules.t5 is available."
                )

        # Load checkpoint from mapping (same pattern as TAEHV/TAESDV; missing path fails in torch.load)
        checkpoint_path = VIDEOVAEPLUS_CHECKPOINT_MAP[model_type]
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        self.load_state_dict(state_dict, strict=False)

    def forward(
        self,
        pixel_values: torch.Tensor,
        captions: Optional[Union[str, List[str]]] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Forward pass. Caption-guided models (videovaeplus_16zcap, videovaeplus_4zcap) use captions;
        others ignore them.

        Args:
            pixel_values (torch.Tensor): Input, shape [B, C, T, H, W] (BCTHW), range [-1, 1].
            captions (Optional[Union[str, List[str]]]): Optional; used only for
                videovaeplus_16zcap/videovaeplus_4zcap. None => text-free fallback with warning.
            **kwargs: Ignored; API compatibility.

        Returns:
            torch.Tensor: Reconstructed video, shape [B, C, T, H, W] (BCTHW), range [-1, 1].
        """
        if self.caption_guide:
            text_embeddings = None
            text_attn_mask = None
            if captions is None:
                warnings.warn(
                    "caption_guide=True but captions is None; "
                    "falling back to text-free reconstruction.",
                    UserWarning,
                )
            else:
                if isinstance(captions, str):
                    captions = [captions]
                text_embeddings, text_attn_mask = (
                    self.text_embedder.get_text_embeddings(captions)
                )
                text_embeddings = text_embeddings.to(
                    dtype=next(self.parameters()).dtype
                )
                text_attn_mask = text_attn_mask.to(
                    dtype=next(self.parameters()).dtype
                )
            z, _ = self.encode(
                pixel_values,
                text_embeddings=text_embeddings,
                text_attn_mask=text_attn_mask,
            )
            dec = self.decode(
                z,
                text_embeddings=text_embeddings,
                text_attn_mask=text_attn_mask,
            )
        else:
            z, _ = self.encode(pixel_values)
            dec = self.decode(z)
        return dec.clamp_(-1, 1)
