"""
Counterfactual Intervention Pipeline (CFI)

Given a real video batch "real_videos", CFI builds a paired counterfactual
batch "cf_videos". The design goal is to change generation-related (causal)
signals while keeping non-causal appearance/context cues aligned to the real clip.

Flow: Intervention - VAEPool randomly picks one model from the configured video VAE pool
and reconstructs the clip;
Prediction - "frequency_domain_alignment" (rFFT magnitude fusion) then "pixel_domain_alignment"
(alpha blend) yields "cf_videos". The real/counterfactual pair is intended to
share non-causal cues while differing on causal content.

Components: VideoReconstructor (one video VAE from vae_pool, inference-only),
VAEPool (random routing over a video VAE pool, optional GPU load/release),
CFIPipeline (reconstruct -> freq -> pixel -> outputs).
"""

import random
from typing import Dict, List, Optional, Type, Union

import torch
import torch.nn as nn

from vae_pool import TAEHV, TAESDV, VideoVAEPlus


def frequency_domain_alignment(
    real_videos: torch.Tensor,
    rec_videos: torch.Tensor,
    freq_fusion_lambda: Optional[float] = None,
) -> torch.Tensor:
    """
    Align reconstruction to real in the frequency domain via 3D rFFT magnitude fusion
    (extended from single-frame practice to full video).

    CFI prediction step (frequency): align high-frequency magnitude to "real_videos"
    while preserving phase from "rec_videos". This keeps non-causal frequency cues
    close to the real clip and preserves reconstruction-specific structure for later
    counterfactual synthesis.

    Uses 3D real FFT (rfftn) over (T, H, W). Only the amplitude spectrum is fused:
    high-freq bins use real_videos amplitude, low-freq use rec_videos; phase is
    preserved from rec_videos everywhere to avoid ghosting. The freq_fusion_lambda
    in [0, 0.5] is the fraction of bins (by radial frequency) treated as high-freq.
    Scheduling: None => uniformly sampled in [0, 0.5] per call; <= 0 => no replacement
    (return rec_videos as-is).

    FFT/quantile/IFFT are executed explicitly in float32 inside this function
    because torch.fft.rfftn may reject lower-precision inputs (for example,
    bfloat16 under autocast in some environments). The final aligned video is
    cast back to the original input dtype; irfftn output is real.

    Args:
        real_videos (torch.Tensor): Real video, shape (B, T, C, H, W).
        rec_videos (torch.Tensor): Reconstructed video, same shape and device/dtype.
        freq_fusion_lambda (Optional[float]): Fraction of spectrum [0, 0.5] as
            high-freq (quantile). None => sample in [0, 0.5]; <= 0 => no replacement.

    Returns:
        torch.Tensor: Aligned video (far_videos) same shape/dtype/device as inputs.
    """
    # Independent of pixel_fusion_lambda: None => sample [0, 0.5]; <= 0 => skip.
    if freq_fusion_lambda is None:
        freq_fusion_lambda = random.uniform(0.0, 0.5)
    if freq_fusion_lambda <= 0.0:
        return rec_videos
    _, T, _, H, W = real_videos.shape
    device = real_videos.device
    dtype = real_videos.dtype

    # Execute frequency-domain alignment in float32 for torch.fft compatibility
    # under AMP/bfloat16; the final real output is cast back to input dtype.
    real_fft = torch.fft.rfftn(real_videos.float(), dim=(-4, -2, -1))  # (B, T, C, H, W//2+1)
    rec_fft = torch.fft.rfftn(rec_videos.float(), dim=(-4, -2, -1))    # (B, T, C, H, W//2+1)

    # Radial frequency r per bin (half spectrum layout)
    ft = torch.fft.fftfreq(T, device=device).abs()   # (T,)
    fh = torch.fft.fftfreq(H, device=device).abs()   # (H,)
    fw = torch.fft.rfftfreq(W, device=device)        # (W//2+1,)
    r = torch.sqrt(
        ft.view(-1, 1, 1).pow(2)
        + fh.view(1, -1, 1).pow(2)
        + fw.view(1, 1, -1).pow(2)
    )  # (T, H, W//2+1)
    r_max = r.max().clamp(min=1e-8)
    freq_norm = r / r_max  # (T, H, W//2+1)

    # quantile() requires float/double; freq_norm is built from
    # fftfreq/rfftfreq (default float), so it is already float/double here.
    threshold = torch.quantile(freq_norm.flatten(), 1.0 - freq_fusion_lambda)
    # high_freq_mask: (1, T, 1, H, W//2+1), broadcastable to (B, T, C, H, W//2+1)
    high_freq_mask = (freq_norm >= threshold).unsqueeze(0).unsqueeze(2)

    # Amplitude-only replacement; phase always from rec_videos
    real_amp = torch.abs(real_fft)   # (B, T, C, H, W//2+1)
    rec_amp = torch.abs(rec_fft)     # (B, T, C, H, W//2+1)
    rec_pha = torch.angle(rec_fft)   # (B, T, C, H, W//2+1)
    fused_amp = torch.where(high_freq_mask, real_amp, rec_amp)  # (B, T, C, H, W//2+1)
    aligned_fft = torch.polar(fused_amp, rec_pha)         # (B, T, C, H, W//2+1)

    # frequency-aligned reconstructed videos (far_videos, B, T, C, H, W)
    far_videos = torch.fft.irfftn(
        aligned_fft, s=(T, H, W), dim=(-4, -2, -1)
    )
    return far_videos.to(dtype=dtype)


def pixel_domain_alignment(
    real_videos: torch.Tensor,
    far_videos: torch.Tensor,
    pixel_fusion_lambda: Optional[float] = None,
) -> torch.Tensor:
    """
    CFI prediction step (spatial): alpha-style fusion between "real_videos" and
    "far_videos" to produce "cf_videos".

    "cf_videos" = pixel_fusion_lambda*real_videos + (1-pixel_fusion_lambda)*far_videos.
    Scheduling is independent of "frequency_domain_alignment" (when both use "None",
    each step draws its own value). None => uniform in [0, 0.5] per call here; <= 0 => no
    real injection (return far_videos as-is).

    Args:
        real_videos (torch.Tensor): Real video (B, T, C, H, W).
        far_videos (torch.Tensor): Frequency-aligned reconstructed video, same shape.
        pixel_fusion_lambda (Optional[float]): Fusion weight in [0, 0.5]. None =>
            sample in [0, 0.5]; <= 0 => no fusion (return far_videos).

    Returns:
        torch.Tensor: Counterfactual batch "cf_videos", same shape/dtype/device.
    """
    # Independent of freq_fusion_lambda: None => sample [0, 0.5]; <= 0 => skip.
    if pixel_fusion_lambda is None:
        pixel_fusion_lambda = random.uniform(0.0, 0.5)
    if pixel_fusion_lambda <= 0.0:
        return far_videos
    lam = torch.tensor(
        pixel_fusion_lambda,
        device=real_videos.device,
        dtype=real_videos.dtype,
    ).view(1, 1, 1, 1, 1)
    cf_videos = lam * real_videos + (1.0 - lam) * far_videos
    return cf_videos


class VideoReconstructor(nn.Module):
    """
    Unified video reconstruction model (inference-only) for one architecture from the
    video VAE pool (vae_pool package: TAEHV, TAESDV, or VideoVAEPlus).

    Used in CFI intervention: the reconstruction step perturbs generation-related
    content before later fusion aligns non-causal cues with the real input. Provides a single
    interface: input and output are [B, T, C, H, W] (BTCHW) with ImageNet normalization
    for supported vae_pool models. CFIPipeline asserts reconstruction shape matches input
    before frequency/pixel fusion; standalone use assumes the underlying VAE preserves shape.
    Internally, value range and layout are converted per-model convention:
    - TAEHV / TAESDV: NTCHW (same as BTCHW here), value range [0, 1].
    - VideoVAEPlus: BCTHW, value range [-1, 1].

    Every model in the video VAE pool is invoked with the same signature:
    model(converted_input, captions=captions).
    Caption is only used by videovaeplus_16zcap/videovaeplus_4zcap; TAEHV, TAESDV, and
    other non-caption models accept but ignore it via **kwargs.

    Args:
        model_type (str): One of (shapes/ranges follow each class in vae_pool):
            TAEHV "taecvx", "taehv", "taew2_1", "taew2_2", "taehv1_5", "taeltx_2", "taeltx2_3",
            "taeltx2_3_wide", "taeos1_3"; TAESDV "taesdv"; VideoVAEPlus "videovaeplus_16z",
            "videovaeplus_4z", "videovaeplus_16zcap", "videovaeplus_4zcap".

    Callers can always pass captions; videovaeplus_16zcap/videovaeplus_4zcap use them, others ignore via **kwargs.
    """

    # model_type -> class for the video VAE pool (VideoVAEPlus: videovaeplus_* series).
    MODEL_CLASSES: Dict[str, Type[nn.Module]] = {
        "taecvx": TAEHV,
        "taehv": TAEHV,
        "taew2_1": TAEHV,
        "taew2_2": TAEHV,
        "taehv1_5": TAEHV,
        "taeltx_2": TAEHV,
        "taeltx2_3": TAEHV,
        "taeltx2_3_wide": TAEHV,
        "taeos1_3": TAEHV,
        "taesdv": TAESDV,
        "videovaeplus_16z": VideoVAEPlus,
        "videovaeplus_4z": VideoVAEPlus,
        "videovaeplus_16zcap": VideoVAEPlus,
        "videovaeplus_4zcap": VideoVAEPlus,
    }

    # For reference only: videovaeplus_16zcap/videovaeplus_4zcap use captions; others ignore via **kwargs.
    CAPTION_MODELS = {"videovaeplus_16zcap", "videovaeplus_4zcap"}
    # NTCHW in [0, 1]: no BCTHW permute; same branch as TAEHV for ImageNet range conversion.
    NTCHW_ZERO_ONE_MODELS = {
        "taecvx",
        "taehv",
        "taew2_1",
        "taew2_2",
        "taehv1_5",
        "taeltx_2",
        "taeltx2_3",
        "taeltx2_3_wide",
        "taeos1_3",
        "taesdv",
    }

    def __init__(self, model_type: str = "taehv") -> None:
        """
        Initialize the video reconstruction model (inference-only).

        Args:
            model_type (str): Model type; see class docstring for options.
        """
        super().__init__()
        if model_type not in self.MODEL_CLASSES:
            raise ValueError(
                f"Unsupported model_type: {model_type}. "
                f"Choose from: {list(self.MODEL_CLASSES.keys())}"
            )
        self.model_type = model_type

        model_class = self.MODEL_CLASSES[model_type]
        self.model = model_class(model_type=model_type)

        # ImageNet mean/std as buffers so they follow .to(device) and state_dict.
        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32),
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32),
        )

    # Pipeline convention: external interface uses ImageNet-normalized BTCHW.
    # x_norm = (x_01 - mean) / std (mean/std per channel, C=3); buffers imagenet_mean/std.
    # Helpers below convert to/from [0,1] or [-1,1] per selected architecture.

    def _denormalize_imagenet(self, x: torch.Tensor) -> torch.Tensor:
        """
        Denormalize ImageNet-normalized tensor back to [0, 1] pixel range.

        ImageNet normalization is: x_norm = (x_01 - mean) / std. This method
        inverts it: x_01 = x_norm * std + mean, then clamps to [0, 1] so that
        values stay in valid pixel range (e.g. after rounding or model output).

        Args:
            x (torch.Tensor): ImageNet-normalized tensor, shape [B, T, C, H, W], C=3.

        Returns:
            torch.Tensor: Tensor in [0, 1] range, same shape.
        """
        # (1,1,3,1,1) broadcasts over [B, T, C, H, W]; per-channel mean/std from buffers.
        mean = self.imagenet_mean.to(device=x.device, dtype=x.dtype).view(1, 1, 3, 1, 1)
        std = self.imagenet_std.to(device=x.device, dtype=x.dtype).view(1, 1, 3, 1, 1)
        x_denorm = x * std + mean
        return x_denorm.clamp(0.0, 1.0)

    def _normalize_imagenet(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize tensor from [0, 1] pixel range to ImageNet normalization.

        Applies x_norm = (x_01 - mean) / std so outputs match the same BTCHW ImageNet
        convention as this module's inputs and forward outputs.

        Args:
            x (torch.Tensor): Tensor in [0, 1] range, shape [B, T, C, H, W], C=3.

        Returns:
            torch.Tensor: ImageNet-normalized tensor, same shape.
        """
        # (1,1,3,1,1) broadcasts over [B, T, C, H, W]; per-channel mean/std from buffers.
        mean = self.imagenet_mean.to(device=x.device, dtype=x.dtype).view(1, 1, 3, 1, 1)
        std = self.imagenet_std.to(device=x.device, dtype=x.dtype).view(1, 1, 3, 1, 1)
        return (x - mean) / std

    def _imagenet_to_model_range(
        self, x: torch.Tensor, target_range: str
    ) -> torch.Tensor:
        """
        Convert ImageNet-normalized input to the range expected by the VAE.

        First we denormalize to [0, 1]. Then:
        - target_range "0_1": TAEHV and similar; no further change.
        - target_range "-1_1": VideoVAEPlus; linear map [0,1] -> [-1,1].

        Args:
            x (torch.Tensor): ImageNet-normalized tensor, shape [B, T, C, H, W].
            target_range (str): "0_1" or "-1_1".

        Returns:
            torch.Tensor: Tensor in the requested range, same shape.
        """
        x_01 = self._denormalize_imagenet(x)
        if target_range == "0_1":
            return x_01
        if target_range == "-1_1":
            return x_01 * 2.0 - 1.0
        raise ValueError(f"Unsupported target_range: {target_range}")

    def _model_range_to_imagenet(
        self, x: torch.Tensor, source_range: str
    ) -> torch.Tensor:
        """
        Convert VAE output from model range to ImageNet normalization.

        VAE output is either [0, 1] (TAEHV) or [-1, 1] (VideoVAEPlus).
        We clamp to the nominal range, map to [0, 1] if needed, then apply
        ImageNet normalization so the result is consistent with the pipeline.

        Args:
            x (torch.Tensor): Tensor in model output range. In this module always
                [B, T, C, H, W] (BTCHW) as produced by the forward branches.
            source_range (str): "0_1" or "-1_1".

        Returns:
            torch.Tensor: ImageNet-normalized tensor, same shape as x.
        """
        if source_range == "0_1":
            x_01 = x.clamp(0.0, 1.0)
        elif source_range == "-1_1":
            x_01 = ((x + 1.0) / 2.0).clamp(0.0, 1.0)
        else:
            raise ValueError(f"Unsupported source_range: {source_range}")
        return self._normalize_imagenet(x_01)

    def forward(
        self,
        pixel_values: torch.Tensor,
        captions: Optional[Union[str, List[str]]] = None,
    ) -> torch.Tensor:
        """
        Reconstruct video: input and output are BTCHW, ImageNet-normalized.

        Value range and layout are converted internally to match each underlying model:
        - TAEHV / TAESDV: input/output [0, 1], layout NTCHW (same as BTCHW here).
        - VideoVAEPlus: input/output [-1, 1], layout BCTHW (permute from BTCHW).

        Every model in the video VAE pool is called as model(converted_input, captions=captions).
        Caption is only used by videovaeplus_16zcap/videovaeplus_4zcap; TAEHV, TAESDV, and others
        ignore it via **kwargs.

        Args:
            pixel_values (torch.Tensor): Video [B, T, C, H, W], ImageNet-normalized.
            captions (Optional[Union[str, List[str]]]): Optional; single string or
                list. Used only by videovaeplus_16zcap/videovaeplus_4zcap.

        Returns:
            torch.Tensor: Reconstructed video [B, T, C, H, W], ImageNet-normalized.
        """
        if self.model_type in self.NTCHW_ZERO_ONE_MODELS:
            # TAEHV / TAESDV: NTCHW [0, 1]. Layout already BTCHW; only value range conversion.
            x_01 = self._imagenet_to_model_range(pixel_values, target_range="0_1")
            reconstructions = self.model(x_01, captions=captions)
            return self._model_range_to_imagenet(reconstructions, source_range="0_1")
        # VideoVAEPlus: BCTHW [-1, 1].
        x_m11 = self._imagenet_to_model_range(pixel_values, target_range="-1_1")
        reconstructions = (
            self.model(x_m11.permute(0, 2, 1, 3, 4).contiguous(), captions=captions)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
        )
        return self._model_range_to_imagenet(reconstructions, source_range="-1_1")


class VAEPool(nn.Module):
    """
    Video VAE pool: multiple VideoReconstructor instances for memory-efficient routing.

    All models in the pool stay on CPU until a forward pass. Each forward randomly selects
    one reconstructor, moves it to pixel_values.device for inference, then (by default)
    moves it back to CPU so only one video VAE occupies GPU memory at a time.

    Args:
        model_types (List[str]): model_type list; each entry valid for VideoReconstructor.
        release_after_use (bool): If True (default), move the selected model back to CPU
            after each forward to free GPU memory. Set False to keep the last-used model on
            device (e.g. for repeated calls with the same reconstructor).
    """

    def __init__(
        self,
        model_types: List[str],
        release_after_use: bool = True,
    ) -> None:
        super().__init__()
        if not model_types:
            raise ValueError("model_types must be a non-empty list.")
        self._reconstructors = nn.ModuleList(
            [VideoReconstructor(model_type=mt) for mt in model_types]
        )
        for rec in self._reconstructors:
            rec.to("cpu")
        self._model_types = list(model_types)
        self._release_after_use = release_after_use

    def forward(
        self,
        pixel_values: torch.Tensor,
        captions: Optional[Union[str, List[str]]] = None,
    ) -> torch.Tensor:
        """
        Reconstruct using one randomly chosen model from the video VAE pool.

        The selected VideoReconstructor is moved to pixel_values.device for inference,
        then back to CPU if release_after_use is True.

        Args:
            pixel_values (torch.Tensor): Video [B, T, C, H, W], ImageNet-normalized.
            captions (Optional[Union[str, List[str]]]): Optional; used only by videovaeplus_16zcap/videovaeplus_4zcap.

        Returns:
            torch.Tensor: Reconstructed video [B, T, C, H, W], ImageNet-normalized.
        """
        idx = torch.randint(0, len(self._reconstructors), (1,), device="cpu").item()
        rec = self._reconstructors[idx]
        rec.to(pixel_values.device)
        try:
            return rec(pixel_values, captions=captions)
        finally:
            if self._release_after_use:
                rec.to("cpu")


class CFIPipeline(nn.Module):
    """
    Counterfactual Intervention Pipeline ("CFIPipeline"): video VAE reconstruction then
    frequency- and pixel-domain fusion so "cf_videos" is paired with "real_videos".

    Chain: VAEPool (random model from the video VAE pool) -> reconstruction ->
    "frequency_domain_alignment"
    -> "pixel_domain_alignment".

    Input: real videos (ImageNet-normalized BTCHW) and optional captions.
    Output: dict with "real_videos", "rec_videos", "far_videos", "cf_videos"
    (pixel step: cf_videos = lambda*real_videos + (1-lambda)*far_videos).

    Forward asserts rec_videos.shape == real_videos.shape before fusion so temporal
    mismatches fail fast (e.g. misconfigured VAE).
    """

    def __init__(
        self,
        model_types: List[str],
        freq_fusion_lambda: Optional[float] = None,
        pixel_fusion_lambda: Optional[float] = None,
        release_after_use: bool = True,
    ) -> None:
        """
        Args:
            model_types (List[str]): model_type list for the video VAE pool (random routing).
            freq_fusion_lambda (Optional[float]): (T,H,W) spectrum fraction [0, 0.5]
                to inject real magnitude (high-freq). None => uniform sample in
                [0, 0.5] per forward (independent of pixel_fusion_lambda). 0 => no injection.
            pixel_fusion_lambda (Optional[float]): Pixel fusion weight [0, 0.5].
                None => uniform sample in [0, 0.5] per forward (independent of
                freq_fusion_lambda). cf_videos =
                pixel_fusion_lambda*real_videos + (1-pixel_fusion_lambda)*far_videos.
            release_after_use (bool): Whether to move the selected model back to CPU after each forward.
        """
        super().__init__()
        self.vae_pool = VAEPool(
            model_types=model_types,
            release_after_use=release_after_use,
        )
        self.freq_fusion_lambda = freq_fusion_lambda
        self.pixel_fusion_lambda = pixel_fusion_lambda

    def forward(
        self,
        pixel_values: torch.Tensor,
        captions: Optional[Union[str, List[str]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Run full CFI: intervention (video VAE reconstruction via self.vae_pool) then prediction
        (frequency + pixel fusion) to form "cf_videos" paired with "real_videos".

        Args:
            pixel_values (torch.Tensor): Real videos [B, T, C, H, W], ImageNet-normalized.
            captions (Optional[Union[str, List[str]]]): Optional; used only by
                videovaeplus_16zcap/videovaeplus_4zcap.

        Returns:
            Dict[str, torch.Tensor]: "real_videos", "rec_videos", "far_videos",
                "cf_videos", all [B, T, C, H, W], ImageNet-normalized, for further
                modules in the same convention.
        """
        real_videos = pixel_values
        rec_videos = self.vae_pool(real_videos, captions=captions)
        assert rec_videos.shape == real_videos.shape, (
            f"CFI rec_videos shape {tuple(rec_videos.shape)} != real_videos "
            f"{tuple(real_videos.shape)}."
        )
        far_videos = frequency_domain_alignment(
            real_videos, rec_videos, freq_fusion_lambda=self.freq_fusion_lambda
        )
        cf_videos = pixel_domain_alignment(
            real_videos, far_videos, pixel_fusion_lambda=self.pixel_fusion_lambda
        )
        return {
            "real_videos": real_videos,
            "rec_videos": rec_videos,
            "far_videos": far_videos,
            "cf_videos": cf_videos,
        }
