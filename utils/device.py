"""
Single source of truth for resolving the device a run should use.
All entry points (training, sampling, evaluation) must obtain their
torch.device from resolve_device() rather than constructing one directly.
"""
import torch

from config.config import ExperimentConfig


def resolve_device(cfg: ExperimentConfig) -> torch.device:
    """Return the torch.device for this run, with a clear warning if CUDA
    was requested but is unavailable."""
    if cfg.device == "cuda" and not torch.cuda.is_available():
        print(
            "WARNING: cfg.device='cuda' but CUDA is not available on this machine. "
            "Falling back to CPU — training will be significantly slower. "
            "For a quick CPU smoke test, consider a smaller config "
            "(e.g. fewer epochs, smaller model, and fewer generated samples)."
        )
        return torch.device("cpu")
    return torch.device(cfg.device)
