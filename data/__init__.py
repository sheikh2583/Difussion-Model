"""
data — dataset loading and preprocessing.

Primary entry point (used by runner.py / train.py):
    get_dataloaders_for_config(cfg: ExperimentConfig)
        Routes to the correct dataset loader based on cfg.dataset.name.
        Supports: "cifar10" (32×32), "celeba" (configurable image_size).
        To add a new dataset, create data/<name>.py and add one
        @_register entry in data/dataset_registry.py — nothing else changes.

Legacy CIFAR-10 direct exports (kept for backward compatibility):
    get_dataloaders(cfg, batch_size, seed)  — from data.cifar10
    get_datasets(cfg)                        — from data.cifar10
"""

from data.cifar10 import get_dataloaders, get_datasets
from data.dataset_registry import get_dataloaders_for_config, DATASET_REGISTRY

__all__ = [
    # Primary multi-dataset interface
    "get_dataloaders_for_config",
    "DATASET_REGISTRY",
    # Legacy cifar10 direct interface
    "get_dataloaders",
    "get_datasets",
]
