"""
dataset_registry.py — single dispatch point for all datasets.

Instead of importing cifar10.get_dataloaders directly everywhere,
all pipeline code imports this module.  Adding a new dataset is one
entry in DATASET_REGISTRY — nothing else changes.

Supported dataset names (cfg.dataset.name):
    "cifar10"  — CIFAR-10 32×32 (original)
    "celeba"   — CelebA, center-cropped and resized to cfg.dataset.image_size
    "celeba_latent" — verified normalized 4x16x16 CelebA latent cache
"""

from config.config import DatasetConfig, ExperimentConfig

DATASET_REGISTRY = {}


def _register(name):
    def decorator(fn):
        DATASET_REGISTRY[name] = fn
        return fn
    return decorator


@_register("cifar10")
def _cifar10(cfg: DatasetConfig, batch_size: int, seed: int):
    from data.cifar10 import get_dataloaders
    return get_dataloaders(cfg, batch_size, seed)


@_register("celeba")
def _celeba(cfg: DatasetConfig, batch_size: int, seed: int):
    from data.celeba import get_dataloaders
    return get_dataloaders(cfg, batch_size, seed)


@_register("celeba_latent")
def _celeba_latent(cfg: DatasetConfig, batch_size: int, seed: int):
    from data.celeba_latent import get_dataloaders
    return get_dataloaders(cfg, batch_size, seed)


def get_dataloaders_for_config(cfg: ExperimentConfig):
    """
    Primary entry point used by runner.py and train.py.
    Routes to the correct dataset loader based on cfg.dataset.name.
    """
    name = cfg.dataset.name
    if name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{name}'. "
            f"Available: {list(DATASET_REGISTRY.keys())}. "
            f"To add a new dataset: create data/<name>.py mirroring cifar10.py, "
            f"then add one @_register entry in data/dataset_registry.py."
        )
    return DATASET_REGISTRY[name](cfg.dataset, cfg.batch_size, cfg.seed)
