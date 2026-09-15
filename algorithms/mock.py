"""
MockAlgorithm — a purely functional smoke test of the shared pipeline.

It is NOT an approximation, simplification, or stand-in for Flow
Matching or Mean Flow. It uses trivial, unrelated math (a plain
autoencoding-style L2 objective and a dummy iterative refinement loop)
solely to exercise every interface the pipeline requires:
  - training_step returns a real scalar loss with grad;
  - sample respects n_samples / nfe / device and returns correctly
    shaped images;
  - checkpointing, timing, logging, evaluation, and plotting all get
    real (if meaningless) numbers to operate on.
"""
from typing import Any, Dict

import torch

from algorithms.base import BaseAlgorithm


class MockAlgorithm(BaseAlgorithm):
    def training_step(self, batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        device = batch.device
        # Arbitrary scalar "time" input required by the shared backbone's
        # signature — has no algorithmic meaning here.
        t = torch.rand(batch.shape[0], device=device)
        pred = self.model(batch, t)
        loss = torch.nn.functional.mse_loss(pred, batch)
        return {"loss": loss}

    @torch.no_grad()
    def sample(self, n_samples: int, nfe: int, device: torch.device) -> torch.Tensor:
        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")
        if nfe < 1:
            raise ValueError(f"nfe must be >= 1, got {nfe}")

        image_shape = (n_samples, self.model.cfg.in_channels,
                        _infer_image_size(self.model), _infer_image_size(self.model))
        x = torch.randn(image_shape, device=device)
        # Dummy multi-step "refinement" loop purely to exercise the NFE
        # interface end-to-end. No algorithmic claim is made about it.
        for step in range(nfe):
            t = torch.full((n_samples,), step / nfe, device=device)
            x = self.model(x, t)
        return x.clamp(-1.0, 1.0)


def _infer_image_size(model: torch.nn.Module) -> int:
    # The shared backbone is resolution-agnostic (fully convolutional),
    # so MockAlgorithm needs the image size from wherever the caller
    # configured it. Sampler passes it in via a model attribute set at
    # construction time (see training/trainer.py / sampling/sampler.py).
    return getattr(model, "_expected_image_size", 32)
