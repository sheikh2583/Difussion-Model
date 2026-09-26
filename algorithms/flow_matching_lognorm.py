"""
Flow Matching with Logit-Normal Time Sampling.

Variant of FlowMatchingAlgorithm (Lipman et al. 2022 / rectified-flow style)
that replaces uniform time sampling with logit-normal sampling
(Esser et al. 2024, Stable Diffusion 3).  This concentrates training
on intermediate timesteps where the model learns the most.

The only difference from FlowMatchingAlgorithm is the time distribution
in ``training_step``.  Sampling (Euler ODE integration) is identical
and inherited from the base class.

Pixel tensors are in [-1, 1]; latent presets use normalized codec tensors.
"""

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.flow_matching import FlowMatchingAlgorithm


class FlowMatchingLognormAlgorithm(FlowMatchingAlgorithm):
    """
    Conditional Flow Matching (CFM) with logit-normal time sampling.

    Inherits the Euler ODE sampler from FlowMatchingAlgorithm; only
    the training-time distribution is overridden.

    algorithm_kwargs recognised (all optional):
        logit_mean (float, default 0.0): mean of the Gaussian before sigmoid.
        logit_std (float, default 1.0): positive standard deviation of that
            Gaussian.
    """

    def __init__(self, model: nn.Module, algorithm_kwargs: Dict[str, Any] = None):
        super().__init__(model, algorithm_kwargs)
        self.logit_mean = float(self.algorithm_kwargs.get("logit_mean", 0.0))
        self.logit_std = float(self.algorithm_kwargs.get("logit_std", 1.0))
        if self.logit_std <= 0:
            raise ValueError(f"logit_std must be positive, got {self.logit_std}")

    # ------------------------------------------------------------------
    # Training (overrides uniform time sampling with logit-normal)
    # ------------------------------------------------------------------

    def training_step(self, batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        One gradient step of the Flow Matching objective with logit-normal
        time sampling instead of uniform.

        Steps identical to FlowMatchingAlgorithm.training_step except Step 2:
            2. Sample t ~ sigmoid(logit_mean + logit_std * N(0,1))
               instead of t ~ Uniform(0, 1).
        """
        x_data = batch  # real images in [-1, 1], shape (B, C, H, W)

        # --- Step 1: sample Gaussian noise with the same shape as x_data ---
        epsilon = torch.randn_like(x_data)  # epsilon ~ N(0, I)

        # --- Step 2: logit-normal time sampling (Esser et al. 2024) ---
        u = (
            torch.randn(x_data.shape[0], device=x_data.device) * self.logit_std
            + self.logit_mean
        )
        t = torch.sigmoid(u)

        # --- Step 3: compute the noisy interpolant x_t ---
        t_view = t.view(-1, 1, 1, 1)
        x_t = (1.0 - t_view) * x_data + t_view * epsilon

        # --- Step 4: compute the ground-truth velocity ---
        target_v = epsilon - x_data  # shape (B, C, H, W)

        # --- Step 5: predict velocity with the backbone ---
        pred_v = self.model(x_t, t)  # shape (B, C, H, W)

        # --- Step 6: flow matching loss ---
        loss = F.mse_loss(pred_v, target_v)  # scalar

        return {"loss": loss}

    # sample() is inherited from FlowMatchingAlgorithm — identical Euler ODE.
