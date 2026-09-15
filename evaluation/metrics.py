"""
FID / Inception Score computation, decoupled from which algorithm
produced the images. Uses torchmetrics so the same reference
implementation is used for every algorithm's evaluation.

Images are fed through the InceptionV3 feature extractor in fixed-size
batches rather than all at once: a single forward pass over thousands
of images allocates memory for the whole batch simultaneously (worse
on CPU, which has no incremental/streaming allocator the way batched
GPU kernels do), and can exceed available RAM/VRAM long before the
metric math itself would ever need that much memory at once. Batch
size is an implementation/performance detail only — it does not change
the resulting FID/IS values, so it is fixed here rather than exposed
as an experimental control.

IMPORTANT: FrechetInceptionDistance accumulates its real/fake feature
statistics as plain Python lists internally, which are NOT captured by
`nn.Module.state_dict()`/`load_state_dict()` (only registered
tensors/buffers are). Caching a metric object's `state_dict()` and
reloading it therefore silently produces a metric with zero
accumulated samples — no error at save or load time, but a
"more than one sample is required" crash at compute() time. To avoid
this, the real-image *tensor* is cached/loaded (not the metric
object), and a fresh FrechetInceptionDistance is constructed for every
evaluation call, updated with both real and fake images within that
same call.
"""
import os
from typing import Optional, Tuple

import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

_METRIC_BATCH_SIZE = 100


def _to_uint8(images: torch.Tensor) -> torch.Tensor:
    """images expected in [-1, 1]; torchmetrics wants uint8 in [0, 255]."""
    images = (images.clamp(-1, 1) + 1) / 2.0
    return (images * 255).to(torch.uint8)


def _batched_update(metric, images: torch.Tensor, device: torch.device, **update_kwargs) -> None:
    for start in range(0, images.shape[0], _METRIC_BATCH_SIZE):
        chunk = images[start:start + _METRIC_BATCH_SIZE]
        metric.update(_to_uint8(chunk).to(device), **update_kwargs)


def cache_real_images(real_images: torch.Tensor, cache_path: str) -> None:
    """Persist the raw real-image reference set (CPU tensor) to disk."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(real_images.cpu(), cache_path)


def load_real_images(cache_path: str, device: torch.device) -> torch.Tensor:
    # PyTorch >=2.6 defaults torch.load to weights_only=True; safe to disable
    # here since we only ever load tensors this project itself wrote.
    return torch.load(cache_path, map_location=device, weights_only=False)


def compute_fid(real_images: torch.Tensor, generated_images: torch.Tensor,
                 device: torch.device) -> float:
    """
    Builds a fresh FID metric, updates it with both the real reference
    images and the generated images, and computes the score in one
    call. real_images is expected to already be on `device` (or movable
    to it); it is NOT a pre-built metric object (see module docstring
    for why that caching approach was unreliable).
    """
    fid = FrechetInceptionDistance(normalize=False).to(device)
    _batched_update(fid, real_images, device, real=True)
    _batched_update(fid, generated_images, device, real=False)
    return fid.compute().item()


def compute_inception_score(generated_images: torch.Tensor,
                             device: torch.device) -> Tuple[float, float]:
    inception = InceptionScore(normalize=False).to(device)
    _batched_update(inception, generated_images, device)
    mean, std = inception.compute()
    return mean.item(), std.item()
