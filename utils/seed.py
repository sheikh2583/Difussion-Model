"""
utils.seed — reproducible randomness helpers.

set_seed(seed)
    Seeds Python's built-in `random` module, NumPy, and PyTorch (both
    CPU and all CUDA devices) from a single integer.  Called once at
    the beginning of Trainer.fit() so that weight initialisation,
    data-loader shuffling, noise sampling, and dropout are all
    deterministically reproducible given the same seed.

set_deterministic(deterministic)
    Toggles cuDNN's deterministic mode and disables the auto-tuner
    (benchmark=False) when determinism is requested.  Deterministic
    mode eliminates non-deterministic cuDNN algorithms at the cost of
    some throughput; it is enabled by default during training so that
    results are exactly reproducible on the same GPU.
    Setting deterministic=False re-enables the benchmark auto-tuner
    for maximum throughput when exact reproducibility is not required.
"""
import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_deterministic(deterministic: bool = True) -> None:
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
