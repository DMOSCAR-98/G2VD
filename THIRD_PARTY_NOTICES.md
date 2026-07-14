# Third-Party Notices

This repository contains project-owned G2VD code and code adapted from,
wrapped around, or inspired by prior open-source research implementations.
Project-owned code is released under the Apache License 2.0 unless otherwise
noted. Third-party code, model weights, and datasets remain subject to their
original licenses and attribution requirements.

If you use this repository, please cite the G2VD paper and the relevant
third-party methods/datasets used in your experiments.

## Baseline Implementations

The following baseline implementations are adapted from the public implementation
associated with:

- Chen et al., "DeMamba: AI-generated video detection on million-scale GenVideo benchmark".
- Project repository: https://github.com/chenhaoxing/DeMamba
- License reported by the upstream project: Apache-2.0.

This applies to the reproduced detector baselines in `baseline_models/` other
than the Hugging Face based backbone wrappers listed below, including:

- F3Net
- FTCN
- MINTIME
- NPR
- STIL
- TALL

The corresponding academic methods should also be cited when those baselines are
used:

- F3Net: "Thinking in Frequency: Face Forgery Detection by Mining Frequency-Aware Clues".
- FTCN: "Exploring Temporal Coherence for More General Video Face Forgery Detection".
- MINTIME: "MINTIME: Multi-Identity Size-Invariant Video Deepfake Detection".
- NPR: "Rethinking the Up-Sampling Operations in CNN-based Generative Network for Generalizable Deepfake Detection".
- STIL: "Spatiotemporal Inconsistency Learning for DeepFake Video Detection".
- TALL: "TALL: Thumbnail Layout for Deepfake Video Detection".

## Hugging Face / Backbone Components

The following models are instantiated through Hugging Face Transformers or
Hugging Face compatible model repositories:

- CLIP
- XCLIP
- TimeSformer
- VideoMAE
- ViViT

Please follow the licenses and usage terms of the corresponding upstream model
repositories and cite the original papers:

- CLIP: "Learning Transferable Visual Models From Natural Language Supervision".
- XCLIP: "Expanding Language-Image Pretrained Models for General Video Recognition".
- TimeSformer: "Is Space-Time Attention All You Need for Video Understanding?"
- VideoMAE: "VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training".
- ViViT: "ViViT: A Video Vision Transformer".

## DeMamba Components

The DeMamba-style CLIP/XCLIP variants in `video_backbones/` build on the
DeMamba design and related Mamba implementations:

- DeMamba project: https://github.com/chenhaoxing/DeMamba
- Mamba reference implementation noted in source files: https://github.com/alxndrTL/mamba.py
- Additional minimal Mamba reference noted in source files: https://github.com/johnma2006/mamba-minimal

## CLIP Tokenizer/Model Utilities

`baseline_models/clip/` contains CLIP utility code and vocabulary resources
derived from the public CLIP implementation. Please follow the original CLIP
license and attribution terms.

## VAE Components

`vae_pool/` contains wrappers and source components for lightweight temporal
autoencoders and VideoVAE+ variants used by the counterfactual intervention
pipeline. Source files include upstream references where applicable, including:

- TAE / TAEHV / TAESDV components from madebyollin repositories:
  - TAESD: https://github.com/madebyollin/taesd
  - TAESDV: https://github.com/madebyollin/taesdv
  - TAEHV: https://github.com/madebyollin/taehv
  - The TAEHV repository is released under the MIT License; follow the upstream
    license and weight terms for all TAE-family components.
- VideoVAE+ components:
  - Paper: "Large Motion Video Autoencoding with Cross-modal Video VAE".
  - arXiv: https://arxiv.org/abs/2412.17805
  - Project/code: https://github.com/VideoVerses/VideoVAEPlus
  - Upstream repository license: CC-BY-NC-ND; follow the upstream terms before
    using or redistributing VideoVAE+-related code or weights.
- Utility functions adapted from diffusion-model codebases where noted in the
  source files.

Please follow the corresponding upstream licenses and cite the relevant papers
when using those components.

The root Apache-2.0 license does not override the licenses or usage terms of
these VAE components or their pretrained weights.

## Dataset References

The code uses metadata for public AI-generated video detection datasets. The
video data themselves are not redistributed in this repository. Please obtain
each dataset from its official release channel and cite the corresponding paper:

- GenVidBench: "GenVidBench: A 6-Million Benchmark for AI-Generated Video Detection".
- GenVideo: "DeMamba: AI-generated video detection on million-scale GenVideo benchmark".
- GVD: "AI-Generated Video Detection via Spatial-Temporal Anomaly Learning".
- GVF: "DeCoF: Generated Video Detection via Frame Consistency: The First Benchmark Dataset".

## Reporting Missing Attribution

If you find missing or incomplete attribution, please open an issue or contact
the maintainers so the notice can be corrected.
