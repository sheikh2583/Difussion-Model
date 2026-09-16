"""
Future Direction 02: Multi-Scale MeanFlow for High-Resolution Generation.

SLIDE REFERENCE: "Coarse-to-Fine Cascade" — thesis presentation, future work.

Background
----------
Training a MeanFlow model directly at high resolution is expensive.  A
coarse-to-fine strategy reduces cost by:
  1. Generating a semantically coherent low-resolution (16×16) image cheaply
     using a small MeanFlow model (fewer parameters, smaller batch cost).
  2. Upsampling and using it as a warm initialisation z_init for a 32×32 model,
     replacing pure noise.  Starting from z_init at some intermediate time
     t_start < 1.0 reduces the effective NFE needed at the fine stage.

This is analogous to the SDEdit / img2img paradigm but formulated entirely
within the MeanFlow displacement-identity framework, avoiding any solver changes.

Planned Approach
----------------
Stage 1 — Coarse generation (16×16):
    z_coarse = coarse_algorithm.sample(n_samples, coarse_nfe, device)
    # shape: (B, C, 16, 16), values in [-1, 1]

Stage 2 — Upsample to fine resolution:
    z_init = F.interpolate(z_coarse, size=self.fine_size, mode='bilinear',
                           align_corners=False)
    # shape: (B, C, 32, 32)

Stage 3 — Fine refinement from z_init:
    # Instead of starting at t=1 (pure noise), inject z_init at t=t_start.
    # The fine model then applies fine_nfe displacement steps from t_start → 0.
    # t_start ≈ 0.5 means the fine model only needs to correct mid-frequency
    # artefacts introduced by bilinear upsampling.
    times = torch.linspace(t_start, 0.0, fine_nfe + 1, device=device)
    z = z_init.clone()
    for i in range(fine_nfe):
        t_b = torch.full((B,), times[i].item(), device=device)
        r_b = torch.full((B,), times[i+1].item(), device=device)
        u   = fine_algorithm._forward(z, r_b, t_b)
        z   = z - (times[i] - times[i+1]) * u
    return z.clamp(-1, 1)

Research Question
-----------------
Does the coarse-to-fine cascade preserve MeanFlow's low-NFE advantage at 32×32
compared to training a 32×32 model from scratch?
Expected outcome: at total NFE = coarse_nfe + fine_nfe = 2 + 1 = 3, this
pipeline should approach the FID of a 5-step single-scale 32×32 model.

Known Prerequisites Before Implementing
-----------------------------------------
1. DatasetConfig needs a `scale_factor` field (e.g. 0.5 → 16×16 output)
   so that the coarse training run sees downscaled images automatically.
   Currently DatasetConfig only supports image_size; a transform pipeline
   change is required in data/cifar10.py.
2. Two separate training runs needed:
   a. mf_coarse_16.json   → trains 16×16 MeanFlow model.
   b. mf_full.json        → existing 32×32 MeanFlow model (no change needed).
3. Sampler.sample() needs an optional z_init parameter; currently it always
   allocates fresh torch.randn noise.  This requires a one-line change in
   sampling/sampler.py to accept and pass through z_init.
4. t_start is a hyperparameter that trades off diversity vs. refinement
   fidelity.  Should be tuned on a small grid (0.25, 0.5, 0.75).

NOT a BaseAlgorithm subclass.  No training_step defined here.  Both
coarse and fine algorithms must already be trained and checkpointed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from algorithms.mean_flow import MeanFlowAlgorithm


class MultiScaleMeanFlowPipeline:
    """
    Coarse-to-fine cascade sampling wrapper for MeanFlow.

    Generates images in two stages:
      1. Coarse model produces a low-resolution draft.
      2. Fine model refines from the upsampled draft via the displacement
         identity, starting at an intermediate time t_start rather than
         pure noise (t = 1).

    Parameters
    ----------
    coarse_algorithm : MeanFlowAlgorithm
        A MeanFlow model trained at `coarse_size` × `coarse_size`.
        Must expose .sample() and ._forward().
    fine_algorithm : MeanFlowAlgorithm
        A MeanFlow model trained at `fine_size` × `fine_size`.
        Must expose ._forward().
    coarse_size : int
        Spatial resolution of the coarse model output.  Default: 16.
    fine_size : int
        Target spatial resolution after refinement.  Default: 32.
    t_start : float
        Time at which the upsampled coarse image is injected into the
        fine model's trajectory.  Values in (0, 1); lower values mean
        more conservative refinement (fewer artefacts, less diversity).
        Default: 0.5.
    """

    def __init__(
        self,
        coarse_algorithm: "MeanFlowAlgorithm",
        fine_algorithm:   "MeanFlowAlgorithm",
        coarse_size: int   = 16,
        fine_size:   int   = 32,
        t_start:     float = 0.5,
    ):
        self.coarse      = coarse_algorithm
        self.fine        = fine_algorithm
        self.coarse_size = coarse_size
        self.fine_size   = fine_size
        self.t_start     = t_start

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample(
        self,
        n_samples:  int,
        coarse_nfe: int,
        fine_nfe:   int,
        device:     torch.device,
    ) -> torch.Tensor:
        """
        Run the coarse-to-fine cascade.

        Parameters
        ----------
        n_samples  : Number of images to generate.
        coarse_nfe : NFE budget for the coarse stage.
        fine_nfe   : NFE budget for the fine refinement stage.
        device     : Target device for all tensors.

        Returns
        -------
        torch.Tensor
            Shape (n_samples, C, fine_size, fine_size), clamped to [-1, 1].
        """
        raise NotImplementedError(
            "TODO:\n"
            "\n"
            "Step 1 — Coarse generation:\n"
            "  z_coarse = self.coarse.sample(n_samples, coarse_nfe, device)\n"
            "  # z_coarse: (B, C, coarse_size, coarse_size)\n"
            "\n"
            "Step 2 — Upsample to fine resolution:\n"
            "  z_init = F.interpolate(\n"
            "      z_coarse, size=self.fine_size,\n"
            "      mode='bilinear', align_corners=False\n"
            "  )  # (B, C, fine_size, fine_size)\n"
            "\n"
            "Step 3 — Fine refinement from z_init at t=t_start:\n"
            "  times = torch.linspace(self.t_start, 0.0, fine_nfe + 1, device=device)\n"
            "  z = z_init.clone()\n"
            "  B = z.shape[0]\n"
            "  with torch.no_grad():\n"
            "      for i in range(fine_nfe):\n"
            "          t_b = torch.full((B,), times[i].item(),     device=device)\n"
            "          r_b = torch.full((B,), times[i+1].item(),   device=device)\n"
            "          u   = self.fine._forward(z, r_b, t_b)\n"
            "          z   = z - (times[i] - times[i+1]) * u\n"
            "\n"
            "  return z.clamp(-1.0, 1.0)\n"
            "\n"
            "PREREQUISITE: Sampler.sample() must accept an optional z_init kwarg.\n"
            "PREREQUISITE: A 16x16-trained MeanFlow checkpoint must exist."
        )
