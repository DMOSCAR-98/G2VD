"""
Video VAE pool: three video VAE architectures in one package.

- taehv.py: Hunyuan-style tiny video VAE; multiple model_type presets.
- taesdv.py: Stable Diffusion Videos-style tiny video VAE.
- videovaeplus.py: unified VideoVAEPlus variants; implementation code lives under
  videovaeplus_src/ (vendored modules, no separate public API).

Each class documents input/output layout and range; constructors preserve dimensions
where stated. Checkpoint locations are defined only in each module's checkpoint map.
"""

from .taehv import TAEHV
from .taesdv import TAESDV
from .videovaeplus import VideoVAEPlus

__all__ = [
    "TAEHV",
    "TAESDV",
    "VideoVAEPlus",
]
