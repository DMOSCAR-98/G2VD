"""Global RNG seeding for reproducible training and evaluation."""
import random

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Fix Python random, NumPy, and PyTorch RNG state in the current process.

    Call after merging CLI into cfg and before data_sampler, create_dataloaders,
    or model construction. DataLoader subprocesses re-seed in create_dataloaders
    when num_workers > 0 (worker_init_fn and cfg seed).

    Does not enable full CUDA determinism; use torch.use_deterministic_algorithms
    separately if required (may reduce performance).

    Args:
        seed (int): Integer seed from merged cfg (YAML/CLI), for example
            provided as `seed=42` on the command line.

    Note:
        Callers should pass cfg seed as already resolved to int (OmegaConf/YAML
        types); this function does not coerce non-int values.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
