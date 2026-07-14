#!/usr/bin/env python3
# Upstream: https://github.com/madebyollin/taehv
# Synced from upstream commit (branch main, file taehv.py):
#   a1c8e6a2ba77b91f284ef98935ec5bd21a41d786
#   https://github.com/madebyollin/taehv/blob/a1c8e6a2ba77b91f284ef98935ec5bd21a41d786/taehv.py
#
# Local changes on top of upstream:
# - TAEHV_CHECKPOINT_MAP + model_type (default "taehv"); paths centralized in the map,
#   keyed only by model_type (not ad-hoc path strings). taeltx2_3_wide matches branch
#   2026_03_11_taeltx23_wide (wide decoder).
# - decode_video: no leading trim (unlike upstream); forward crops decoded time length to the
#   original input frame count so encode tail padding does not leak to outputs.
# - forward: show_progress_bar default False; **kwargs ignored.
# - StreamingTAEHV and main() removed.

"""
Tiny AutoEncoder in the Hunyuan Video architecture
(DNN for encoding / decoding videos to Hunyuan Video's latent space).

Temporal: encode_video may pad the time axis to a multiple of t_downscale; decode_video returns
RGB with one frame per step on that padded timeline. forward crops to the caller's original frame
count T_in (see decode_video). t_downscale is larger for LTX model_types (taeltx_2, taeltx2_3,
taeltx2_3_wide) than for default Hunyuan-style presets.

Spatial: H and W should be compatible with self.patch_size and the encoder's three stride-2
spatial stages (for patch_size==1 presets, multiples of 8 are typically sufficient).
"""
from collections import namedtuple
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

TWorkItem = namedtuple("TWorkItem", ("input_tensor", "block_index"))

TAEHV_CHECKPOINT_MAP: Dict[str, str] = {
    "taecvx": "./pretrained_weights/vae_pool/taehv/taecvx.pth",
    "taehv": "./pretrained_weights/vae_pool/taehv/taehv.pth",
    "taew2_1": "./pretrained_weights/vae_pool/taehv/taew2_1.pth",
    "taew2_2": "./pretrained_weights/vae_pool/taehv/taew2_2.pth",
    "taehv1_5": "./pretrained_weights/vae_pool/taehv/taehv1_5.pth",
    "taeltx_2": "./pretrained_weights/vae_pool/taehv/taeltx_2.pth",
    "taeltx2_3": "./pretrained_weights/vae_pool/taehv/taeltx2_3.pth",
    "taeltx2_3_wide": "./pretrained_weights/vae_pool/taehv/taeltx2_3_wide.pth",
    "taeos1_3": "./pretrained_weights/vae_pool/taehv/taeos1_3.pth",
}


def conv(n_in, n_out, **kwargs):
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)


class Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 3) * 3


class MemBlock(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Sequential(
            conv(n_in * 2, n_out),
            nn.ReLU(inplace=True),
            conv(n_out, n_out),
            nn.ReLU(inplace=True),
            conv(n_out, n_out),
        )
        self.skip = (
            nn.Conv2d(n_in, n_out, 1, bias=False)
            if n_in != n_out
            else nn.Identity()
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, past):
        return self.act(self.conv(torch.cat([x, past], 1)) + self.skip(x))


class WideMemBlock(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        groups = max(1, n_out // 64)
        assert n_out % groups == 0, f"{n_out} % {groups} ??"
        self.conv = nn.Sequential(
            nn.Conv2d(n_in * 2, n_out, 1),
            nn.ReLU(inplace=True),
            conv(n_out, n_out, groups=groups),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_out, n_out, 1),
            nn.ReLU(inplace=True),
            conv(n_out, n_out, groups=groups),
        )
        self.skip = (
            nn.Conv2d(n_in, n_out, 1, bias=False)
            if n_in != n_out
            else nn.Identity()
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, past):
        return self.act(self.conv(torch.cat([x, past], 1)) + self.skip(x))


class TPool(nn.Module):
    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f * stride, n_f, 1, bias=False)

    def forward(self, x):
        _NT, C, H, W = x.shape
        return self.conv(x.reshape(-1, self.stride * C, H, W))


class TGrow(nn.Module):
    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f, n_f * stride, 1, bias=False)

    def forward(self, x):
        _NT, C, H, W = x.shape
        x = self.conv(x)
        return x.reshape(-1, C, H, W)


def apply_model_with_memblocks_parallel(model, x, show_progress_bar):
    """
    Apply a sequential model with memblocks to the given input,
    with parallelization over the time axis and iteration over blocks.

    Args:
    - model: nn.Sequential of blocks to apply
    - x: input data, of dimensions NTCHW
    - show_progress_bar: if True, enables tqdm progressbar display

    Returns NTCHW tensor of output data.
    """
    assert x.ndim == 5, f"TAEHV operates on NTCHW tensors, but got {x.ndim}-dim tensor"
    N, T, C, H, W = x.shape
    x = x.reshape(N * T, C, H, W)

    # parallel over input timesteps, iterate over blocks
    for b in tqdm(model, disable=not show_progress_bar):
        if isinstance(b, (MemBlock, WideMemBlock)):
            NT, C, H, W = x.shape
            T = NT // N
            _x = x.reshape(N, T, C, H, W)
            # pad with zeros along time axis (i.e. empty memory), slice
            block_memory = F.pad(_x, (0, 0, 0, 0, 0, 0, 1, 0), value=0)[
                :, :T
            ].reshape(x.shape)
            x = b(x, block_memory)
        else:
            x = b(x)
    NT, C, H, W = x.shape
    T = NT // N
    return x.view(N, T, C, H, W)


def apply_model_with_memblocks_sequential_single_step(
    model, memory, work_queue, progress_bar=None
):
    """
    Process the work queue (a graph traversal over blocks and timesteps)
    until an output frame is produced or the queue is empty.
    Mutates memory and work_queue in place.

    Returns N1CHW output tensor, or None if the queue needs more input.
    """
    while work_queue:
        xt, i = work_queue.pop(0)
        if progress_bar is not None and i == 0:
            progress_bar.update(1)
        if i == len(model):
            return xt.unsqueeze(1)
        b = model[i]
        if isinstance(b, (MemBlock, WideMemBlock)):
            # mem blocks are simple since we're visiting the graph in causal order
            if memory[i] is None:
                xt_new = b(xt, xt * 0)
            else:
                xt_new = b(xt, memory[i])
            memory[i] = xt
            work_queue.insert(0, TWorkItem(xt_new, i + 1))
        elif isinstance(b, TPool):
            # pool blocks accumulate inputs until they have enough to pool
            if memory[i] is None:
                memory[i] = []
            memory[i].append(xt)
            if len(memory[i]) > b.stride:
                raise ValueError(
                    f"TPool memory overflow: {len(memory[i])} items for stride {b.stride}"
                )
            elif len(memory[i]) == b.stride:
                N, C, H, W = xt.shape
                xt = b(torch.cat(memory[i], 1).view(N * b.stride, C, H, W))
                memory[i] = []
                work_queue.insert(0, TWorkItem(xt, i + 1))
        elif isinstance(b, TGrow):
            xt = b(xt)
            NT, C, H, W = xt.shape
            for xt_next in reversed(
                xt.view(NT // b.stride, b.stride * C, H, W).chunk(b.stride, 1)
            ):
                work_queue.insert(0, TWorkItem(xt_next, i + 1))
        else:
            xt = b(xt)
            work_queue.insert(0, TWorkItem(xt, i + 1))
    return None


def apply_model_with_memblocks_sequential(model, x, show_progress_bar):
    """
    Apply a sequential model with memblocks to the given input,
    with iteration over timesteps as well as blocks.

    Args:
    - model: nn.Sequential of blocks to apply
    - x: input data, of dimensions NTCHW
    - show_progress_bar: if True, enables tqdm progressbar display

    Returns NTCHW tensor of output data.
    """
    assert x.ndim == 5, f"TAEHV operates on NTCHW tensors, but got {x.ndim}-dim tensor"
    work_queue = [TWorkItem(xt, 0) for xt in x.unbind(1)]
    memory = [None] * len(model)
    progress_bar = tqdm(range(len(work_queue)), disable=not show_progress_bar)
    out = []
    while work_queue:
        xt = apply_model_with_memblocks_sequential_single_step(
            model, memory, work_queue, progress_bar
        )
        if xt is not None:
            out.append(xt)
    progress_bar.close()
    return torch.cat(out, 1)


def apply_model_with_memblocks(model, x, parallel, show_progress_bar):
    """
    Apply a sequential model with memblocks to the given input.
    Args:
    - model: nn.Sequential of blocks to apply
    - x: input data, of dimensions NTCHW
    - parallel: if True, parallelize over timesteps (fast but uses O(T) memory)
        if False, each timestep will be processed sequentially (slow but uses O(1) memory)
    - show_progress_bar: if True, enables tqdm progressbar display

    Returns NTCHW tensor of output data.
    """
    if parallel:
        return apply_model_with_memblocks_parallel(model, x, show_progress_bar)
    else:
        return apply_model_with_memblocks_sequential(model, x, show_progress_bar)


class TAEHV(nn.Module):
    def __init__(
        self,
        model_type: str = "taehv",
        encoder_time_downscale=(True, True, False),
        decoder_time_upscale=(False, True, True),
        decoder_space_upscale=(True, True, True),
        patch_size=1,
        latent_channels=16,
    ):
        """Initialize pretrained TAEHV from TAEHV_CHECKPOINT_MAP[model_type].

        Variant-specific layout (patch size, latents, temporal pool flags) follows upstream rules
        but is keyed only by model_type (e.g. taew2_2, taehv1_5, taeltx_2, taeltx2_3,
        taeltx2_3_wide), not by the weight file path string.

        Args:
            model_type: key into TAEHV_CHECKPOINT_MAP.
            encoder_time_downscale: whether temporal downsampling is enabled for each block.
            decoder_time_upscale: whether temporal upsampling is enabled for each block.
            decoder_space_upscale: whether spatial upsampling is enabled for each block.
            patch_size: input/output pixelshuffle patch-size for this model.
            latent_channels: number of latent channels (z dim) for this model.
        """
        super().__init__()
        self.model_type = model_type
        if model_type not in TAEHV_CHECKPOINT_MAP:
            raise ValueError(
                f"Unsupported model_type: {model_type}. "
                f"Choose from: {list(TAEHV_CHECKPOINT_MAP.keys())}"
            )
        checkpoint_path = TAEHV_CHECKPOINT_MAP[model_type]

        self.patch_size = patch_size
        self.latent_channels = latent_channels
        self.image_channels = 3
        if len(decoder_time_upscale) == 2:
            decoder_time_upscale = (False, *decoder_time_upscale)
        if model_type == "taew2_2":
            self.patch_size, self.latent_channels = 2, 48
        elif model_type == "taehv1_5":
            self.patch_size, self.latent_channels = 2, 32
        elif model_type in ("taeltx_2", "taeltx2_3", "taeltx2_3_wide"):
            (
                self.patch_size,
                self.latent_channels,
                encoder_time_downscale,
                decoder_time_upscale,
            ) = (4, 128, (True, True, True), (True, True, True))
        self.encoder = nn.Sequential(
            conv(self.image_channels * self.patch_size ** 2, 64),
            nn.ReLU(inplace=True),
            TPool(64, 2 if encoder_time_downscale[0] else 1),
            conv(64, 64, stride=2, bias=False),
            MemBlock(64, 64),
            MemBlock(64, 64),
            MemBlock(64, 64),
            TPool(64, 2 if encoder_time_downscale[1] else 1),
            conv(64, 64, stride=2, bias=False),
            MemBlock(64, 64),
            MemBlock(64, 64),
            MemBlock(64, 64),
            TPool(64, 2 if encoder_time_downscale[2] else 1),
            conv(64, 64, stride=2, bias=False),
            MemBlock(64, 64),
            MemBlock(64, 64),
            MemBlock(64, 64),
            conv(64, self.latent_channels),
        )
        n_f = [256, 128, 64, 64]
        self.decoder = nn.Sequential(
            Clamp(),
            conv(self.latent_channels, n_f[0]),
            nn.ReLU(inplace=True),
            MemBlock(n_f[0], n_f[0]),
            MemBlock(n_f[0], n_f[0]),
            MemBlock(n_f[0], n_f[0]),
            nn.Upsample(scale_factor=2 if decoder_space_upscale[0] else 1),
            TGrow(n_f[0], 2 if decoder_time_upscale[0] else 1),
            conv(n_f[0], n_f[1], bias=False),
            MemBlock(n_f[1], n_f[1]),
            MemBlock(n_f[1], n_f[1]),
            MemBlock(n_f[1], n_f[1]),
            nn.Upsample(scale_factor=2 if decoder_space_upscale[1] else 1),
            TGrow(n_f[1], 2 if decoder_time_upscale[1] else 1),
            conv(n_f[1], n_f[2], bias=False),
            MemBlock(n_f[2], n_f[2]),
            MemBlock(n_f[2], n_f[2]),
            MemBlock(n_f[2], n_f[2]),
            nn.Upsample(scale_factor=2 if decoder_space_upscale[2] else 1),
            TGrow(n_f[2], 2 if decoder_time_upscale[2] else 1),
            conv(n_f[2], n_f[3], bias=False),
            nn.ReLU(inplace=True),
            conv(n_f[3], self.image_channels * self.patch_size ** 2),
        )
        # Same wide-decoder branch as upstream for model_type taeltx2_3_wide (see file header).
        if model_type == "taeltx2_3_wide":
            n_f = [1024, 512, 256, 64]
            self.decoder = nn.Sequential(
                Clamp(),
                conv(self.latent_channels, n_f[0]),
                nn.ReLU(inplace=True),
                WideMemBlock(n_f[0], n_f[0]),
                WideMemBlock(n_f[0], n_f[0]),
                WideMemBlock(n_f[0], n_f[0]),
                nn.Upsample(scale_factor=2 if decoder_space_upscale[0] else 1),
                TGrow(n_f[0], 2 if decoder_time_upscale[0] else 1),
                conv(n_f[0], n_f[1], bias=False),
                WideMemBlock(n_f[1], n_f[1]),
                WideMemBlock(n_f[1], n_f[1]),
                WideMemBlock(n_f[1], n_f[1]),
                nn.Upsample(scale_factor=2 if decoder_space_upscale[1] else 1),
                TGrow(n_f[1], 2 if decoder_time_upscale[1] else 1),
                conv(n_f[1], n_f[2], bias=False),
                WideMemBlock(n_f[2], n_f[2]),
                WideMemBlock(n_f[2], n_f[2]),
                WideMemBlock(n_f[2], n_f[2]),
                nn.Upsample(scale_factor=2 if decoder_space_upscale[2] else 1),
                TGrow(n_f[2], 2 if decoder_time_upscale[2] else 1),
                conv(n_f[2], n_f[3], bias=False),
                nn.ReLU(inplace=True),
                conv(n_f[3], self.image_channels * self.patch_size ** 2),
            )
        # computed properties
        self.t_downscale = 2 ** sum(
            t.stride == 2 for t in self.encoder if isinstance(t, TPool)
        )
        self.t_upscale = 2 ** sum(
            t.stride == 2 for t in self.decoder if isinstance(t, TGrow)
        )
        # Upstream skip_trim-related fields (commented; see decode_video docstring).
        # self.is_cogvideox = model_type == "taecvx"
        # self.frames_to_trim = self.t_upscale - 1  # leading RGB warm-up frames trimmed upstream

        self.load_state_dict(
            self.patch_tgrow_layers(
                torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            )
        )

    def patch_tgrow_layers(self, sd):
        """Patch TGrow layers to use a smaller kernel if needed.

        Args:
            sd: state dict to patch
        """
        new_sd = self.state_dict()
        for i, layer in enumerate(self.decoder):
            if isinstance(layer, TGrow):
                key = f"decoder.{i}.conv.weight"
                if sd[key].shape[0] > new_sd[key].shape[0]:
                    # take the last-timestep output channels
                    sd[key] = sd[key][-new_sd[key].shape[0] :]
        return sd

    def preprocess_input_frames(self, x):
        """Preprocess RGB input frames prior to the main encoder sequence."""
        if self.patch_size > 1:
            x = F.pixel_unshuffle(x, self.patch_size)
        return x

    def encode_video(self, x, parallel=True, show_progress_bar=True):
        """Encode a sequence of frames.

        Args:
            x: input NTCHW RGB (C=3) tensor with values in [0, 1].
            parallel: if True, all frames will be processed at once.
                (this is faster but may require more memory).
                if False, frames will be processed sequentially.
        Returns NTCHW latent tensor with ~Gaussian values.
        """
        x = self.preprocess_input_frames(x)
        if x.shape[1] % self.t_downscale != 0:
            # pad at end to multiple of self.t_downscale
            n_pad = self.t_downscale - x.shape[1] % self.t_downscale
            padding = x[:, -1:].repeat_interleave(n_pad, dim=1)
            x = torch.cat([x, padding], 1)
        return apply_model_with_memblocks(self.encoder, x, parallel, show_progress_bar)

    def postprocess_output_frames(self, x):
        """Postprocess RGB frames after the main decoder sequence."""
        if self.patch_size > 1:
            x = F.pixel_shuffle(x, self.patch_size)
        return x.clamp_(0, 1)

    def decode_video(self, x, parallel=True, show_progress_bar=True):
        """Decode a sequence of frames.

        Args:
            x: input NTCHW latent (C=self.latent_channels) tensor with ~Gaussian values.
            parallel: if True, all frames will be processed at once.
            show_progress_bar: tqdm in apply_model_with_memblocks.

        Returns:
            NTCHW RGB in approximately [0, 1].

        Time length: With T_in the frame count before encode tail padding and T_pad after, this
        call returns T_dec = T_pad. If T_in is divisible by t_downscale then T_pad = T_in; else
        T_pad > T_in and forward keeps the first T_in frames only, dropping extra decode frames
        from tail padding.

        Causal decoder: the first (t_upscale - 1) RGB outputs are warm-up. Upstream trims them
        (plus taecvx rules) for reference Hunyuan/CogVideoX alignment; we do not, so latents and
        RGB share one padded timeline at the cost of weaker early frames. forward crops to T_in
        only to remove padding tail decode output, not warm-up.

        Typical t_downscale / t_upscale pairs are 4 for Hunyuan-style presets and 8 for LTX
        model_types; warm-up length is t_upscale - 1.
        """
        x = apply_model_with_memblocks(self.decoder, x, parallel, show_progress_bar)
        x = self.postprocess_output_frames(x)
        return x

    def forward(
        self,
        x,
        parallel=True,
        show_progress_bar=False,
        **kwargs,
    ):
        """
        Round-trip reconstruction: encode then decode, then crop time to the input length.

        Args:
            x (torch.Tensor): Input NTCHW RGB (C=3) tensor with values in [0, 1].
            parallel (bool): If True, process all frames at once (faster, more memory).
            show_progress_bar (bool): If True, enables tqdm in encode/decode. Default False.
            **kwargs: Ignored (e.g. captions for API compatibility).

        Returns:
            torch.Tensor: NTCHW RGB in [0, 1] with the same time length as the input (same frame
            count as x on the time dimension). When that length is not a multiple of t_downscale,
            trailing decode frames that correspond only to encode tail padding are discarded.
        """
        t_in = x.shape[1]
        z = self.encode_video(x, parallel=parallel, show_progress_bar=show_progress_bar)
        out = self.decode_video(z, parallel=parallel, show_progress_bar=show_progress_bar)
        return out[:, :t_in]
