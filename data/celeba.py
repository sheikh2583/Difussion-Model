"""
CelebA 64×64 pipeline — mirrors cifar10.py interface exactly.

Auto-downloads via torchvision (requires ~1.4GB, downloads once to cfg.root).
If torchvision download fails (Google Drive quota issues are common), set
cfg.dataset.root to a directory containing the manually downloaded
img_align_celeba/ folder from https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

Preprocessing:
  - Center-crop to 178×178 (removes chin/forehead, keeps face)
  - Resize to image_size × image_size (default 64)
  - Normalize to [-1, 1] (same convention as cifar10.py)

All image tensors are in [-1, 1] project-wide.
"""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config.config import DatasetConfig


def build_transforms(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # → [-1, 1]
    ])


def get_datasets(cfg: DatasetConfig):
    tfm = build_transforms(cfg.image_size)
    # split="train" gives ~162k images; split="valid" gives ~20k
    train_set = datasets.CelebA(
        root=cfg.root, split="train", download=True, transform=tfm)
    test_set = datasets.CelebA(
        root=cfg.root, split="valid", download=True, transform=tfm)
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
