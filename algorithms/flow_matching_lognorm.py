"""
Flow Matching algorithm — Lipman et al. 2022 / rectified-flow style
conditional flow matching, adapted for unconditional CIFAR-10 generation.

Mathematical background
-----------------------
We define a probability path from data (t=0) to noise (t=1):
    x_t = (1 - t) * x_data + t * epsilon,    epsilon ~ N(0, I)

The instantaneous velocity along this path (its time-derivative) is:
    v = dx_t/dt = epsilon - x_data

The backbone is trained to predict v given (x_t, t). At sampling time
we integrate the learned ODE backward from t=1 (pure noise) to t=0
(data) using the Euler method with `nfe` steps.

All image tensors are in [-1, 1] (the project-wide convention).
"""

from typing import Any, Dict

import torch
import torch.nn.functional as F

from algorithms.base import BaseAlgorithm


class FlowMatchingLognormAlgorithm(BaseAlgorithm):
    """
    Conditional Flow Matching (CFM) / rectified-flow training and
    Euler ODE sampling on the linear interpolation path between data
    and Gaussian noise.

    algorithm_kwargs recognised (all optional):
        sigma_min (float, default 0.0): unused in the base linear-path
            formulation but reserved for future sigma-min augmentation
            of the interpolation.  Must not shadow any shared config key.
    """

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        One gradient step of the Flow Matching objective.

        Implements (Lipman et al. 2022 / rectified flow):
            1. Sample noise epsilon ~ N(0, I) with the same shape as batch.
            2. Sample a time t ~ Uniform(0, 1) independently per image.
            3. Interpolate: x_t = (1-t)*x_data + t*epsilon.
            4. Ground-truth velocity: v = epsilon - x_data.
            5. Predicted velocity: v_pred = model(x_t, t).
            6. Loss: MSE(v_pred, v).

        Args:
            batch: Real images, shape (B, C, H, W), values in [-1, 1].

        Returns:
            {"loss": scalar tensor with grad attached}.
        """
        x_data = batch  # real images in [-1, 1], shape (B, C, H, W)

        # --- Step 1: sample Gaussian noise with the same shape as x_data ---
        # epsilon is the "pure noise" endpoint of our probability path (t=1)
        epsilon = torch.randn_like(x_data)  # epsilon ~ N(0, I)

        # --- Step 2: sample a time step t ~ Uniform(0, 1), one per image ---
        # t=0 is data, t=1 is noise; we train on random points along the path
        # Logit-normal time sampling (Esser et al. 2024, Stable Diffusion 3)
        # Concentrates training on intermediate timesteps where the model
        # learns the most, rather than uniform sampling.
        u = torch.randn(x_data.shape[0], device=x_data.device)
        t = torch.sigmoid(u)

        # --- Step 3: compute the noisy interpolant x_t ---
        # x_t = (1-t)*x_data + t*epsilon  (linear interpolation between data and noise)
        # reshape t from (B,) to (B, 1, 1, 1) so it broadcasts over (C, H, W)
        t_view = t.view(-1, 1, 1, 1)
        x_t = (1.0 - t_view) * x_data + t_view * epsilon

        # --- Step 4: compute the ground-truth velocity ---
        # v = d/dt [x_t] = epsilon - x_data
        # (points in the direction of increasing t, i.e. data → noise)
        target_v = epsilon - x_data  # shape (B, C, H, W)

        # --- Step 5: predict velocity with the backbone ---
        # model(x_t, t) outputs a vector field of the same shape as x_t;
        # t is passed as a (B,) tensor of scalars in [0, 1]
        pred_v = self.model(x_t, t)  # shape (B, C, H, W)

        # --- Step 6: flow matching loss (MSE between predicted and target velocity) ---
        # Minimising this loss pushes the model to correctly predict the
        # instantaneous velocity at every point along the interpolation path
        loss = F.mse_loss(pred_v, target_v)  # scalar

        return {"loss": loss}

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, n_samples: int, nfe: int, device: torch.device) -> torch.Tensor:
        """
        Generate images by integrating the learned ODE from t=1 to t=0
        using the Euler method with `nfe` equi-spaced steps.

        Euler update rule (one step of size `step` at time t_cur):
            x_{t - step} = x_t - model(x_t, t_cur) * step

        (Subtracting because t is decreasing; the velocity is defined
        to point from data toward noise, so we reverse it to move from
        noise toward data.)

        Args:
            n_samples: Number of images to generate (≥ 1, validated by caller).
            nfe:       Number of function evaluations / Euler steps (≥ 1).
            device:    Target device.

        Returns:
            Tensor of shape (n_samples, C, H, W), clamped to [-1, 1].
        """
        # Determine image shape from backbone configuration
        C = self.model.cfg.in_channels          # number of image channels (3 for CIFAR-10)
        H = W = self.model._expected_image_size  # spatial resolution (32 for CIFAR-10)

        # --- Step 1: start from pure noise at t=1 ---
        x = torch.randn(n_samples, C, H, W, device=device)  # x ~ N(0, I)

        # --- Step 2: define the step size for the Euler discretisation ---
        # We divide the interval [0, 1] into `nfe` equal steps
        step = 1.0 / nfe

        # --- Steps 3–4: Euler ODE integration from t=1 down to t=0 ---
        with torch.no_grad():
            for i in range(nfe):
                # Current time: starts at 1.0, decreases by `step` each iteration
                t_cur = 1.0 - i * step  # scalar float, in (0, 1]

                # Broadcast t_cur to a (B,) tensor as required by model(x, t)
                t_batch = torch.full(
                    (n_samples,), t_cur, device=device, dtype=torch.float32
                )

                # Evaluate the learned velocity field v_theta(x, t)
                v = self.model(x, t_batch)  # shape (n_samples, C, H, W)

                # Euler step: move x in the direction opposite to v (toward t=0)
                # x ← x - v * step   (since t is decreasing, we subtract)
                x = x - v * step

        # --- Step 4: clamp to project back into the valid image range [-1, 1] ---
        # Small numerical drift from Euler integration can push values slightly
        # outside the range; clamping avoids artefacts in downstream metrics
        return x.clamp(-1.0, 1.0)