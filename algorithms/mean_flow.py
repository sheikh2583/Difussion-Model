"""
Mean Flow algorithm — Geng et al. 2025, "Mean Flows for One-Step
Generative Modeling", arXiv:2505.13447.

Mathematical background
-----------------------
Instead of the *instantaneous* velocity v(z_t, t), Mean Flow learns
the *average* velocity over an interval [r, t]:

    u(z_t, r, t)  :=  (1/(t-r)) * integral_r^t  v(z_tau, tau) d_tau

By the displacement identity this satisfies:
    (t - r) * u(z_t, r, t) = z_t - z_r

Differentiating the integral along the trajectory yields the
Mean Flow Identity (the actual training target):

    u(z_t, r, t) = v(z_t, t)  -  (t - r) * d/dt[u(z_t, r, t)]

where the total time derivative is taken along the trajectory:
    d/dt[u] = (∂u/∂z_t)·v  +  ∂u/∂t
            = JVP of u w.r.t. (z_t, r, t) with tangent (v, 0, 1)

At sampling time, the displacement identity lets us jump exactly
from one time to another in a single network call:
    z_r = z_t - (t - r) * u(z_t, r, t)

so with nfe=1 we go from pure noise (t=1) to data (t=0) in one step.

Conditioning on both r and t
------------------------------
The backbone model(x, t) accepts only ONE scalar time per sample.
Mean Flow's u depends on the INTERVAL [r, t], so we need r as well.
Rather than modify the shared backbone (which would break the
FM-vs-MF parameter-count comparison), we introduce a small extra
module self.r_embed that encodes r into a per-channel additive bias,
which is applied to the input before the backbone sees it.  This is
the ONLY sanctioned extension (see project spec).

All image tensors are in [-1, 1] (the project-wide convention).
"""

from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.base import BaseAlgorithm


class MeanFlowAlgorithm(BaseAlgorithm):
    """
    Mean Flow training (MeanFlow Identity + JVP) and few/one-step
    sampling via the displacement identity.

    The network takes (z_t, r, t).  Because the shared backbone only
    accepts one scalar time argument, r is injected by a small learned
    embedding self.r_embed that adds a per-channel spatial bias to the
    input image tensor before the backbone is called.

    NOTE: self.r_embed is an algorithm-inherent module, analogous to an
    algorithm_kwargs entry.  It is distinct from the shared backbone and
    is reported separately as algorithm_extra_parameter_count by the
    pipeline (already wired in results.py / trainer.py).  It does NOT
    alter the backbone's own parameters or architecture, preserving the
    apples-to-apples FM-vs-MF backbone comparison.

    algorithm_kwargs recognised (all optional):
        p_same (float, default 0.25): probability of forcing r == t on
            a given sample, which trains the degenerate/diagonal case
            u(z_t, t, t) = v(z_t, t) explicitly.  This matches the
            original paper's recommendation for training stability.
            Must not shadow any shared config key.
    """

    def __init__(self, model: nn.Module, algorithm_kwargs: Dict[str, Any] = None):
        super().__init__(model, algorithm_kwargs)

        # ----------------------------------------------------------------
        # r-conditioning embedding
        # ----------------------------------------------------------------
        # The shared backbone expects model(x, t) where t is the SINGLE
        # scalar conditioning signal.  Mean Flow's average velocity
        # u(z_t, r, t) depends on BOTH endpoints of the interval [r, t].
        # We encode r into a per-channel additive signal that is broadcast
        # over (H, W) and added to the input image before calling the
        # backbone.  This keeps the backbone architecture unchanged.
        #
        # Architecture: Linear(1 -> 64) -> SiLU -> Linear(64 -> C)
        # Output shape after unsqueezing: (B, C, 1, 1), broadcast → (B, C, H, W)
        C = model.cfg.in_channels  # 3 for CIFAR-10
        self.r_embed = nn.Sequential(
            nn.Linear(1, 64),   # project scalar r to a 64-d feature
            nn.SiLU(),          # non-linearity
            nn.Linear(64, C),   # project down to one value per channel
        )

        # ----------------------------------------------------------------
        # Hyper-parameter: probability of using the diagonal (r == t) case
        # ----------------------------------------------------------------
        # When r == t, (t - r) == 0, so u_tgt = v exactly.  Sampling these
        # cases explicitly during training stabilises convergence.
        self.p_same: float = float(self.algorithm_kwargs.get("p_same", 0.25))

    # ------------------------------------------------------------------
    # Module list — expose r_embed to Trainer/Sampler
    # ------------------------------------------------------------------

    def trainable_modules(self) -> List[nn.Module]:
        """
        Return all trainable components owned by this algorithm.
        The Trainer uses this list (not a hard-coded self.model) for
        optimizer construction, device placement, train/eval mode, and
        checkpointing, so self.r_embed is automatically handled.

        Convention: shared backbone first, algorithm-specific extras after.
        """
        return [self.model, self.r_embed]

    # ------------------------------------------------------------------
    # Internal helper: the conditioned forward pass f(z, r, t)
    # ------------------------------------------------------------------

    def _forward(
        self,
        z: torch.Tensor,       # (B, C, H, W)  noisy image at time t
        r: torch.Tensor,       # (B,)           left endpoint of interval
        t: torch.Tensor,       # (B,)           right endpoint of interval (= current time)
    ) -> torch.Tensor:
        """
        Compute the predicted average velocity u_pred = f(z, r, t).

        Encodes r via self.r_embed and adds it to z as a per-channel
        spatial bias, then passes the conditioned input and t to the
        shared backbone.

        This is the ONLY place where r enters the network; the backbone
        itself is never modified.
        """
        # Encode r: (B,) -> unsqueeze to (B, 1) -> embed to (B, C)
        r_signal = self.r_embed(r.unsqueeze(-1))        # (B, C)
        # Broadcast over spatial dimensions: (B, C) -> (B, C, 1, 1) -> (B, C, H, W)
        x_conditioned = z + r_signal[:, :, None, None]  # additive r-conditioning

        # Call backbone: t is still passed as the explicit time argument
        u_pred = self.model(x_conditioned, t)           # (B, C, H, W)
        return u_pred

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        One gradient step of the Mean Flow objective.

        Implements the MeanFlow Identity (Geng et al. 2025, §3):

            u_tgt = v - (t - r) * d/dt[u]
            loss  = MSE(u_pred, u_tgt.detach())

        where d/dt[u] is the TOTAL time derivative along the trajectory,
        computed via a Jacobian-vector product (JVP) with tangent
        (dz/dt, dr/dt, dt/dt) = (v, 0, 1).

        Args:
            batch: Real images, shape (B, C, H, W), values in [-1, 1].

        Returns:
            {"loss": scalar tensor with grad attached}.
        """
        x_data = batch  # real images in [-1, 1], shape (B, C, H, W)
        B = x_data.shape[0]
        dev = x_data.device

        # --- Step 1: sample noise and compute interpolant z_t ---
        epsilon = torch.randn_like(x_data)              # epsilon ~ N(0, I)

        # Sample t ~ Uniform(0, 1), one per image
        t = torch.rand(B, device=dev)                   # (B,)

        # Compute the ground-truth instantaneous velocity v = d/dt[z_t]
        # v = epsilon - x_data  (direction: data → noise, increasing t)
        v = epsilon - x_data                            # (B, C, H, W)

        # Interpolant at time t: z_t = (1-t)*x_data + t*epsilon
        t_view = t.view(-1, 1, 1, 1)
        z_t = (1.0 - t_view) * x_data + t_view * epsilon  # (B, C, H, W)

        # --- Step 2: sample r ~ Uniform(0, t) with optional diagonal case ---
        # Mean Flow is defined only for r <= t (r is the left endpoint of [r,t]).
        # With probability p_same we set r = t (the diagonal case), which trains
        # u(z_t, t, t) = v(z_t, t) explicitly and stabilises training.
        u_uniform = torch.rand(B, device=dev) * t       # r ~ Uniform(0, t)
        diagonal_mask = torch.rand(B, device=dev) < self.p_same  # True with prob p_same
        r = torch.where(diagonal_mask, t, u_uniform)   # (B,)
        # When r == t, (t - r) == 0, so u_tgt degenerates to v — correct and consistent

        # --- Steps 3–4: JVP to get u_pred and d/dt[u] simultaneously ---
        # We wrap _forward as a functional to differentiate through it.
        # The JVP tangent encodes dz/dt = v (trajectory), dr/dt = 0 (r is fixed
        # w.r.t. t in the identity derivation), dt/dt = 1.
        #
        # torch.func.jvp(f, primals, tangents) returns (f(primals), Jf·tangents)
        # i.e. (u_pred, d/dt[u]) in a single forward-mode AD pass.
        #
        # AMP / autocast caution: JVP through the model under autocast can
        # produce dtype mismatches.  We locally disable autocast for this block.
        # This is a per-algorithm numerical-stability measure, not a change to
        # cfg.amp or any shared file.
        with torch.autocast(device_type=dev.type, enabled=False):
            # Cast primals to float32 to guarantee a consistent dtype during JVP
            z_t_f  = z_t.float()
            r_f    = r.float()
            t_f    = t.float()
            v_f    = v.float()

            # Define the function whose JVP we want:
            #   f(z, r, t) = self._forward(z, r, t)   i.e. u_pred
            # We use torch.func.jvp (functional-transform API, torch >= 2.0)
            u_pred, dudt = torch.func.jvp(
                self._forward,                             # function f
                (z_t_f, r_f, t_f),                        # primals (z, r, t)
                (v_f,                                      # tangent for z: dz/dt = v
                 torch.zeros_like(r_f),                    # tangent for r: dr/dt = 0
                 torch.ones_like(t_f)),                    # tangent for t: dt/dt = 1
            )
            # u_pred : (B, C, H, W) — model's prediction of the average velocity
            # dudt   : (B, C, H, W) — total derivative of u along the trajectory

        # --- Step 5: construct the MeanFlow training target (stop-gradient) ---
        # MeanFlow Identity: u(z_t, r, t) = v - (t - r) * d/dt[u(z_t, r, t)]
        # (t - r) is non-negative by construction; when r == t it is zero,
        # making u_tgt == v, which is the correct degenerate/diagonal case.
        dt = (t_f - r_f).view(-1, 1, 1, 1)             # (B, 1, 1, 1), broadcast-ready
        u_tgt = (v_f - dt * dudt).detach()              # stop-gradient: target does not backprop

        # --- Step 6: Mean Flow loss (MSE between predicted and target average velocity) ---
        loss = F.mse_loss(u_pred, u_tgt)                # scalar

        return {"loss": loss}

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, n_samples: int, nfe: int, device: torch.device) -> torch.Tensor:
        """
        Generate images using the Mean Flow displacement identity in
        `nfe` steps from t=1 (pure noise) to t=0 (data).

        Displacement identity (exact, no further discretisation error):
            z_r = z_t - (t - r) * u(z_t, r, t)

        With nfe=1 this reduces to the paper's headline one-step case:
            z_0 = z_1 - 1.0 * u(z_1, 0, 1)

        Args:
            n_samples: Number of images to generate (≥ 1, validated by caller).
            nfe:       Number of time steps / network evaluations (≥ 1).
            device:    Target device.

        Returns:
            Tensor of shape (n_samples, C, H, W), clamped to [-1, 1].
        """
        # Determine image shape from backbone configuration
        C = self.model.cfg.in_channels          # channels (3 for CIFAR-10)
        H = W = self.model._expected_image_size  # spatial resolution (32)

        # --- Step 1: start from pure noise at t = 1 ---
        z = torch.randn(n_samples, C, H, W, device=device)  # z ~ N(0, I)

        # --- Step 2: build a decreasing sequence of nfe+1 time points ---
        # times[0] = 1.0  (noise),  times[-1] = 0.0  (data)
        times = torch.linspace(1.0, 0.0, nfe + 1, device=device)  # (nfe+1,)

        # --- Step 3: integrate using the displacement identity ---
        with torch.no_grad():
            for i in range(nfe):
                t_cur  = times[i].item()   # left edge of this step (t, larger)
                t_next = times[i + 1].item()  # right edge (r, smaller)

                # Batch scalar tensors for the model call
                t_batch = torch.full((n_samples,), t_cur,  device=device, dtype=torch.float32)
                r_batch = torch.full((n_samples,), t_next, device=device, dtype=torch.float32)

                # Average velocity over the interval [t_next, t_cur]
                # (note: in _forward, arg order is (z, r, t); r=t_next <= t=t_cur)
                u = self._forward(z, r_batch, t_batch)  # (n_samples, C, H, W)

                # Displacement identity: z_r = z_t - (t_cur - t_next) * u
                # This moves z exactly to the image z would be at time t_next,
                # without any further Euler-style discretisation error within the step
                z = z - (t_cur - t_next) * u

        # --- Step 4: clamp to [-1, 1] ---
        # Tiny numerical drift may push values just outside the valid range
        return z.clamp(-1.0, 1.0)