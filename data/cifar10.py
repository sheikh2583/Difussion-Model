"""
Reusable CIFAR-10 pipeline. Both algorithms consume dataloaders built
from this module and nothing else, guaranteeing identical data.
"""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config.config import DatasetConfig


def build_transforms(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # -> [-1, 1]
    ])


def get_datasets(cfg: DatasetConfig):
    tfm = build_transforms(cfg.image_size)
    train_set = datasets.CIFAR10(root=cfg.root, train=True, download=True, transform=tfm)
    test_set = datasets.CIFAR10(root=cfg.root, train=False, download=True, transform=tfm)
    return train_set, test_set


def get_dataloaders(cfg: DatasetConfig, batch_size: int, seed: int):
    train_set, test_set = get_datasets(cfg)

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,
        generator=generator,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader
