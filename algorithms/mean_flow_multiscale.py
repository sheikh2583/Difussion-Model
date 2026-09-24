"""Two-phase iterative refinement for a trained latent Mean Flow model.

Phase 1 creates a low-NFE latent draft. Phase 2 re-noises that draft onto the
training probability path and refines it with the same 3x16x16 latent model.
This is an inference-only wrapper; it is not a training algorithm.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import torch

if TYPE_CHECKING:
    from algorithms.mean_flow import MeanFlowAlgorithm


class MultiScaleMeanFlowPipeline:
    """Coarse-draft then refine sampling with one latent MF model."""

    def __init__(
        self,
        algorithm: "MeanFlowAlgorithm",
        coarse_nfe: int = 1,
        fine_nfe: int = 4,
        t_renoise: float = 0.3,
        codec: Optional[Any] = None,
    ):
        if coarse_nfe < 1:
            raise ValueError(f"coarse_nfe must be >= 1, got {coarse_nfe}")
        if fine_nfe < 1:
            raise ValueError(f"fine_nfe must be >= 1, got {fine_nfe}")
        if not 0.0 <= t_renoise < 1.0:
            raise ValueError(f"t_renoise must be in [0, 1), got {t_renoise}")

        self.algorithm = algorithm
        self.coarse_nfe = coarse_nfe
        self.fine_nfe = fine_nfe
        self.t_renoise = t_renoise
        self.codec = codec

    def sample(self, n_samples: int, device: torch.device) -> torch.Tensor:
        """Generate decoded RGB or unbounded normalized latent samples."""
        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")

        model = self.algorithm.model
        channels = model.cfg.in_channels
        image_size = model._expected_image_size
        if image_size != 16:
            raise ValueError(
                "MultiScaleMeanFlowPipeline is for latent-space models only "
                f"(expected spatial size 16, got {image_size})."
            )

        for module in self.algorithm.trainable_modules():
            module.to(device)
            module.eval()

        with torch.no_grad():
            z = torch.randn(
                n_samples, channels, image_size, image_size, device=device
            )
            coarse_times = torch.linspace(
                1.0, 0.0, self.coarse_nfe + 1, device=device
            )
            for step in range(self.coarse_nfe):
                t_value = float(coarse_times[step])
                r_value = float(coarse_times[step + 1])
                t_batch = torch.full((n_samples,), t_value, device=device)
                r_batch = torch.full((n_samples,), r_value, device=device)
                velocity = self.algorithm._forward(z, r_batch, t_batch)
                z = z - (t_value - r_value) * velocity
            z_coarse = z

            if self.t_renoise > 0.0:
                epsilon = torch.randn_like(z_coarse)
                z = (
                    (1.0 - self.t_renoise) * z_coarse
                    + self.t_renoise * epsilon
                )
            else:
                z = z_coarse

            fine_times = torch.linspace(
                self.t_renoise, 0.0, self.fine_nfe + 1, device=device
            )
            for step in range(self.fine_nfe):
                t_value = float(fine_times[step])
                r_value = float(fine_times[step + 1])
                if t_value == 0.0:
                    break
                t_batch = torch.full((n_samples,), t_value, device=device)
                r_batch = torch.full((n_samples,), r_value, device=device)
                velocity = self.algorithm._forward(z, r_batch, t_batch)
                z = z - (t_value - r_value) * velocity

            if self.codec is not None:
                decoder = getattr(self.codec, "decode_normalised", None)
                if decoder is None:
                    decoder = getattr(self.codec, "decode_normalized", None)
                if decoder is None:
                    raise AttributeError(
                        "codec must expose decode_normalised(normalised_latents)"
                    )
                return decoder(z.to(device)).clamp(-1.0, 1.0)
            return z
