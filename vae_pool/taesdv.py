#!/usr/bin/env python3
# Upstream: https://github.com/madebyollin/taesdv
# Synced from upstream commit (branch main, file taesdv.py):
#   1b2655e57693c2bf2bd016451f86f5cbf5aa5169
#   https://github.com/madebyollin/taesdv/blob/1b2655e57693c2bf2bd016451f86f5cbf5aa5169/taesdv.py
#
# Local changes on top of upstream:
# - TAESDV_CHECKPOINT_MAP + model_type (default "taesdv"); checkpoint path from the map only.
# - forward(x, parallel=..., **kwargs) for round-trip reconstruction; parallel passed to
#   encode_video/decode_video; kwargs ignored.
# - Module docstring: architecture wording (like TAEHV) + note that T is preserved 1:1 (no
#   t_downscale padding); H/W multiples of 8 for spatial alignment.
# - Script entrypoint (main) removed.

"""
Tiny AutoEncoder in the Stable Diffusion Videos architecture
(DNN for encoding / decoding videos to Stable Diffusion video latent space).

There is no temporal downsampling or end-padding on the time axis: encode_video and decode_video
preserve T and per-frame index (same t in and out for round-trip). Spatial H and W should be
multiples of 8 so the three stride-2 spatial encoder stages and matching decoder upsampling stay
aligned (unlike TAEHV's t_downscale rules on T).
"""
from collections import namedtuple
from typing import Dict

import torch
import torch.nn as nn

DecoderResult = namedtuple("DecoderResult", ("frame", "memory"))

TAESDV_CHECKPOINT_MAP: Dict[str, str] = {
    "taesdv": "./pretrained_weights/vae_pool/taesdv/taesdv.pth",
}


def conv(n_in, n_out, **kwargs):
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)


class Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 3) * 3


class Block(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Sequential(
            conv(n_in, n_out),
            nn.ReLU(),
            conv(n_out, n_out),
            nn.ReLU(),
            conv(n_out, n_out),
        )
        self.skip = (
            nn.Conv2d(n_in, n_out, 1, bias=False)
            if n_in != n_out
            else nn.Identity()
        )
        self.fuse = nn.ReLU()

    def forward(self, x):
        return self.fuse(self.conv(x) + self.skip(x))


class MemBlock(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Sequential(
            conv(n_in * 2, n_out),
            nn.ReLU(),
            conv(n_out, n_out),
            nn.ReLU(),
            conv(n_out, n_out),
        )
        self.skip = (
            nn.Conv2d(n_in, n_out, 1, bias=False)
            if n_in != n_out
            else nn.Identity()
        )
        self.act = nn.ReLU()

    def forward(self, x, mem):
        return self.act(self.conv(torch.cat([x, mem], 1)) + self.skip(x))


class TAESDV(nn.Module):
    def __init__(
        self,
        model_type: str = "taesdv",
    ):
        """Initialize pretrained TAESDV.

        Weights are loaded from TAESDV_CHECKPOINT_MAP[model_type].

        Args:
            model_type: key into TAESDV_CHECKPOINT_MAP.
        """
        super().__init__()
        self.model_type = model_type
        if model_type not in TAESDV_CHECKPOINT_MAP:
            raise ValueError(
                f"Unsupported model_type: {model_type}. "
                f"Choose from: {list(TAESDV_CHECKPOINT_MAP.keys())}"
            )
        checkpoint_path = TAESDV_CHECKPOINT_MAP[model_type]

        self.encoder = nn.Sequential(
            conv(3, 64),
            Block(64, 64),
            conv(64, 64, stride=2, bias=False),
            Block(64, 64),
            Block(64, 64),
            Block(64, 64),
            conv(64, 64, stride=2, bias=False),
            Block(64, 64),
            Block(64, 64),
            Block(64, 64),
            conv(64, 64, stride=2, bias=False),
            Block(64, 64),
            Block(64, 64),
            Block(64, 64),
            conv(64, 4),
        )
        self.decoder = nn.Sequential(
            Clamp(),
            conv(4, 64),
            nn.ReLU(),
            MemBlock(64, 64),
            MemBlock(64, 64),
            MemBlock(64, 64),
            nn.Upsample(scale_factor=2),
            conv(64, 64, bias=False),
            MemBlock(64, 64),
            MemBlock(64, 64),
            MemBlock(64, 64),
            nn.Upsample(scale_factor=2),
            conv(64, 64, bias=False),
            MemBlock(64, 64),
            MemBlock(64, 64),
            MemBlock(64, 64),
            nn.Upsample(scale_factor=2),
            conv(64, 64, bias=False),
            MemBlock(64, 64),
            conv(64, 3),
        )
        self.load_state_dict(
            torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        )

    def encode_frame(self, x):
        """Encode a single RGB timestep to latents.

        Args:
            x: input NCHW RGB (C=3) tensor with values in [0, 1].
        Returns NCHW latent tensor with ~Gaussian values.
        """
        assert x.ndim == 4 and x.shape[1] == 3, f"Could not encode frame of shape {x.shape}"
        return self.encoder(x)

    def decode_frame(self, x, mem=None):
        """Decode a single latent timestep to RGB.

        Args:
            x: input NCHW latent (C=4) tensor with ~Gaussian values.
            mem: recurrent memory tensor. Should be:
                None if this is the first decoded frame, or
                memory from previous step if this a subsequent decoded frame.

        Returns a dictionary of:
            frame: NCHW RGB (C=3) decoded video frame with ~[0, 1] values
            memory: memory for decoding subsequent frames.
        """
        assert x.ndim == 4 and x.shape[1] == 4, f"Could not decode frame of shape {x.shape}"
        out_mem, in_mem = [], None if mem is None else list(mem)
        for b in self.decoder:
            if isinstance(b, MemBlock):
                out_mem.append(x)
                x = b(x, x * 0 if in_mem is None else in_mem.pop(0))
            else:
                x = b(x)
        return DecoderResult(x, out_mem)

    def encode_video(self, x, parallel=True):
        """Encode a sequence of frames.

        Args:
            x: input NTCHW RGB (C=3) tensor with values in [0, 1].
            parallel: if True, all frames will be processed at once.
              (this is faster but may require more memory).
              if False, frames will be processed sequentially.
        Returns:
            NTCHW latent (C=4) with ~Gaussian values; same N, T, H, W as input. No temporal
            padding—T matches input frame count. Prefer H and W divisible by 8 (see module docstring).

        """
        assert x.ndim == 5, f"TAESDV operates on NTCHW tensors, but got {x.ndim}-dim tensor"
        N, T, C, H, W = x.shape
        assert C == 3, f"TAESDV encodes RGB tensors, but got {C}-channel tensor"
        if parallel:
            x = self.encode_frame(x.reshape(N * T, C, H, W))
            return x.view(N, T, *x.shape[1:])
        else:
            return torch.stack(
                [
                    self.encode_frame(frame)
                    for frame in x.view(N, T * C, H, W).chunk(T, dim=1)
                ],
                1,
            )

    def decode_video(self, x, parallel=True):
        """Decode a sequence of frames.

        Args:
            x: input NTCHW latent (C=4) tensor with ~Gaussian values.
            parallel: if true, all frames will be processed at once.
              (this is faster but may require more memory).
        Returns:
            NTCHW RGB with ~[0, 1] values; same N, T, H, W as latent input (C=3).

        Temporal/spatial: same T and per-frame alignment as encode_video; H/W as module docstring.
        """
        assert x.ndim == 5, f"TAESDV operates on NTCHW tensors, but got {x.ndim}-dim tensor"
        N, T, C, H, W = x.shape
        assert C == 4, f"TAESDV decodes 4-channel latent tensors, but got {C}-channel tensor"
        if parallel:
            x = x.reshape(N * T, C, H, W)
            for b in self.decoder:
                if isinstance(b, MemBlock):
                    _NT, C, H, W = x.shape
                    # mem is just the current input shifted 1 frame forward along time axis
                    mem = torch.nn.functional.pad(
                        x.reshape(N, T, C, H, W),
                        (0, 0, 0, 0, 0, 0, 1, 0),
                        value=0,
                    )[:, :T].reshape(x.shape)
                    x = b(x, mem)
                else:
                    x = b(x)
            _NT, C, H, W = x.shape
            return x.view(N, T, C, H, W)
        else:
            # if you're running TAESDV in an interactive / real-time loop,
            # this is how you run it.
            out, mem = [], None
            for latent in x.reshape(N, T * C, H, W).chunk(T, dim=1):
                frame, mem = self.decode_frame(latent, mem)
                out.append(frame)
            return torch.stack(out, 1)

    def forward(
        self,
        x,
        parallel=True,
        **kwargs,
    ):
        """Round-trip encode then decode.

        Args:
            x: NTCHW RGB tensor with values in [0, 1].
            parallel: Forwarded to encode_video and decode_video (speed vs memory).
            **kwargs: Ignored (e.g. captions from pipelines).

        Returns:
            NTCHW RGB in ~[0, 1]; same N, T, H, W as input.
        """
        z = self.encode_video(x, parallel=parallel)
        return self.decode_video(z, parallel=parallel)
