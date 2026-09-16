"""Coarse-to-fine sampling pipeline for trained Mean Flow models."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from algorithms.mean_flow import MeanFlowAlgorithm


class MultiScaleMeanFlowPipeline:
    """Generate a coarse draft and refine it with a higher-resolution model."""

    def __init__(
        self,
        coarse_algorithm: "MeanFlowAlgorithm",
        fine_algorithm: "MeanFlowAlgorithm",
        coarse_size: int = 16,
        fine_size: int = 32,
        t_start: float = 0.5,
    ):
        if coarse_size < 1 or fine_size < 1:
            raise ValueError("coarse_size and fine_size must be positive")
        if coarse_size >= fine_size:
            raise ValueError(
                f"coarse_size must be smaller than fine_size, got "
                f"{coarse_size} >= {fine_size}"
            )
        if not 0.0 < t_start <= 1.0:
            raise ValueError(f"t_start must be in (0, 1], got {t_start}")

        self.coarse = coarse_algorithm
        self.fine = fine_algorithm
        self.coarse_size = coarse_size
        self.fine_size = fine_size
        self.t_start = t_start

    def sample(
        self,
        n_samples: int,
        coarse_nfe: int,
        fine_nfe: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Run coarse generation, bilinear upsampling, and fine refinement."""
        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")
        if coarse_nfe < 1 or fine_nfe < 1:
            raise ValueError(
                f"coarse_nfe and fine_nfe must be >= 1, got "
                f"{coarse_nfe} and {fine_nfe}"
            )

        coarse_channels = self.coarse.model.cfg.in_channels
        fine_channels = self.fine.model.cfg.in_channels
        if coarse_channels != fine_channels:
            raise ValueError(
                f"channel mismatch: coarse={coarse_channels}, fine={fine_channels}"
            )

        for algorithm in (self.coarse, self.fine):
            for module in algorithm.trainable_modules():
                module.to(device)
                module.eval()

        with torch.no_grad():
            coarse = self.coarse.sample(n_samples, coarse_nfe, device)
            expected_shape = (
                n_samples,
                coarse_channels,
                self.coarse_size,
                self.coarse_size,
            )
            if tuple(coarse.shape) != expected_shape:
                raise ValueError(
                    f"coarse model returned {tuple(coarse.shape)}, "
                    f"expected {expected_shape}"
                )

            z = F.interpolate(
                coarse,
                size=(self.fine_size, self.fine_size),
                mode="bilinear",
                align_corners=False,
            )
            times = torch.linspace(
                self.t_start, 0.0, fine_nfe + 1, device=device
            )
            for step_index in range(fine_nfe):
                t_value = float(times[step_index])
                r_value = float(times[step_index + 1])
                t_batch = torch.full(
                    (n_samples,), t_value, device=device, dtype=torch.float32
                )
                r_batch = torch.full(
                    (n_samples,), r_value, device=device, dtype=torch.float32
                )
                displacement = self.fine._forward(z, r_batch, t_batch)
                z = z - (t_value - r_value) * displacement

        return z.clamp(-1.0, 1.0)
