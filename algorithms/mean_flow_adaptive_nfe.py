"""Adaptive per-sample NFE allocation for a trained Mean Flow model.

This is an inference wrapper, not a ``BaseAlgorithm`` subclass. It compares
successive clean-image predictions, retires converged samples from the active
batch, and records the realized per-sample and average NFE.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from algorithms.mean_flow import MeanFlowAlgorithm


class AdaptiveMeanFlowSampler:
    """
    Sampling wrapper that allocates NFE per sample based on estimated
    generation difficulty, stopping early when a sample's velocity
    estimate has stabilised.

    Parameters
    ----------
    algorithm : MeanFlowAlgorithm
        A fully trained student (or standard) MeanFlow algorithm.
        Must expose ._forward(z, r, t).
    min_nfe : int
        Minimum number of displacement steps every sample receives
        before it can be deactivated.  Default: 1.
    max_nfe : int
        Hard upper bound on steps.  No sample exceeds this.  Default: 4.
    confidence_threshold : float
        Fractional change in velocity below which a sample is considered
        converged:  ||u_t - u_{t-1}|| / ||u_t|| < threshold.  Default: 0.05.
    """

    def __init__(
        self,
        algorithm: "MeanFlowAlgorithm",
        min_nfe: int = 1,
        max_nfe: int = 4,
        confidence_threshold: float = 0.05,
    ):
        self.algorithm            = algorithm
        self.min_nfe              = min_nfe
        self.max_nfe              = max_nfe
        self.confidence_threshold = confidence_threshold
        self.last_nfe_per_sample: Optional[torch.Tensor] = None
        self.last_average_nfe: Optional[float] = None

        if min_nfe < 1:
            raise ValueError(f"min_nfe must be >= 1, got {min_nfe}")
        if max_nfe < min_nfe:
            raise ValueError(
                f"max_nfe must be >= min_nfe, got {max_nfe} < {min_nfe}"
            )
        if confidence_threshold < 0:
            raise ValueError(
                "confidence_threshold must be non-negative, got "
                f"{confidence_threshold}"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample(self, n_samples: int, device: torch.device) -> torch.Tensor:
        """
        Generate `n_samples` images with adaptive NFE allocation.

        Returns
        -------
        torch.Tensor
            Shape (n_samples, C, H, W), values clamped to [-1, 1].
        """
        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")

        model = self.algorithm.model
        channels = model.cfg.in_channels
        image_size = model._expected_image_size
        z = torch.randn(
            n_samples, channels, image_size, image_size, device=device
        )
        output = torch.empty_like(z)
        previous_prediction = torch.zeros_like(z)
        has_previous = torch.zeros(n_samples, dtype=torch.bool, device=device)
        active = torch.ones(n_samples, dtype=torch.bool, device=device)
        nfe_used = torch.zeros(n_samples, dtype=torch.long, device=device)
        times = torch.linspace(1.0, 0.0, self.max_nfe + 1, device=device)

        for module in self.algorithm.trainable_modules():
            module.eval()

        with torch.no_grad():
            for step_index in range(self.max_nfe):
                indices = active.nonzero(as_tuple=True)[0]
                if indices.numel() == 0:
                    break

                t_value = float(times[step_index])
                next_t = float(times[step_index + 1])
                t_batch = torch.full(
                    (indices.numel(),), t_value, device=device, dtype=torch.float32
                )
                zero_batch = torch.zeros_like(t_batch)

                # Mean Flow predicts average displacement to an arbitrary r.
                # A direct r=0 prediction is a valid clean-image candidate at
                # every step, so early-exited samples never remain at t>0.
                velocity_to_zero = self.algorithm._forward(
                    z[indices], zero_batch, t_batch
                )
                prediction = z[indices] - t_value * velocity_to_zero
                nfe_used[indices] += 1

                can_compare = has_previous[indices]
                difference = (
                    prediction - previous_prediction[indices]
                ).flatten(1).norm(dim=1)
                magnitude = prediction.flatten(1).norm(dim=1).clamp(min=1e-8)
                relative_change = difference / magnitude
                converged = (
                    can_compare
                    & (nfe_used[indices] >= self.min_nfe)
                    & (relative_change < self.confidence_threshold)
                )

                if step_index == self.max_nfe - 1:
                    converged = torch.ones_like(converged)

                finished = indices[converged]
                output[finished] = prediction[converged]
                active[finished] = False

                continuing = ~converged
                if continuing.any():
                    continuing_indices = indices[continuing]
                    previous_prediction[continuing_indices] = prediction[continuing]
                    has_previous[continuing_indices] = True
                    delta_t = t_value - next_t
                    z[continuing_indices] = (
                        z[continuing_indices]
                        - delta_t * velocity_to_zero[continuing]
                    )

        self.last_nfe_per_sample = nfe_used.detach().cpu()
        self.last_average_nfe = float(nfe_used.float().mean().item())
        return output.clamp(-1.0, 1.0)
