"""
Mean Flow algorithm — Geng et al. 2025, arXiv:2505.13447.

Training objective
------------------
Learns the average velocity u(z_t, r, t) over an interval [r, t]:

    u_tgt = v - (t - r) * d/dt[u(z_t, r, t)]

where d/dt[u] is the total time derivative along the trajectory,
approximated by default with a finite difference (FD).  A diagnostic
``use_exact_jvp`` switch restores the original forward-mode AD calculation;
that path runs in fp32 with autocast disabled.

Two-path stochastic training
-----------------------------
Each step randomly routes to one of two paths controlled by p_fd_step:

  FD path  (prob p_fd_step):
      Approximate d/dt[u] via two forward passes with a perturbation
      delta annealed from jvp_delta_start → jvp_delta_end over training.
      Runs under full AMP.  Bias: O(delta).

  Diagonal path  (prob 1 - p_fd_step):
      Force r = t.  The Mean Flow Identity degenerates to u_tgt = v
      exactly.  Single forward pass, zero bias, cheapest possible step.
      This is NOT the same as the p_same diagonal-forcing that happens
      inside the r-sampling — both mechanisms are independent and
      complementary.

algorithm_kwargs
----------------
    p_same          (float, 0.25):  prob of forcing r==t in r-sampling (paper default).
    jvp_delta_start (float, 1e-2):  FD perturbation at epoch 0.
    jvp_delta_end   (float, 1e-4):  FD perturbation at final epoch.
    p_fd_step       (float, 0.5):   prob of taking the FD path per step.
    use_exact_jvp   (bool, False):  replace FD with exact torch.func.jvp on
                                    the detached derivative target path.
    fd_force_fp32   (bool, False):  run the complete finite-difference branch
                                    in fp32 with autocast disabled.
"""

from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import jvp as torch_jvp

from algorithms.base import BaseAlgorithm
from algorithms.r_embed import REmbed


class MeanFlowAlgorithm(BaseAlgorithm):

    def __init__(self, model: nn.Module, algorithm_kwargs: Dict[str, Any] = None):
        super().__init__(model, algorithm_kwargs)

        C = model.cfg.in_channels
        self.r_embed = REmbed(C)

        # --- diagonal-sampling probability (paper §4) ---
        self.p_same: float = float(self.algorithm_kwargs.get("p_same", 0.25))

        # --- FD-JVP hyper-parameters ---
        self.jvp_delta_start: float = float(self.algorithm_kwargs.get("jvp_delta_start", 1e-2))
        self.jvp_delta_end:   float = float(self.algorithm_kwargs.get("jvp_delta_end",   1e-4))
        self.p_fd_step:       float = float(self.algorithm_kwargs.get("p_fd_step",       0.5))
        if self.jvp_delta_start <= 0 or self.jvp_delta_end <= 0:
            raise ValueError("jvp_delta_start and jvp_delta_end must be positive")
        use_exact_jvp = self.algorithm_kwargs.get("use_exact_jvp", False)
        if not isinstance(use_exact_jvp, bool):
            raise TypeError("algorithm_kwargs['use_exact_jvp'] must be a boolean")
        self.use_exact_jvp: bool = use_exact_jvp
        fd_force_fp32 = self.algorithm_kwargs.get("fd_force_fp32", False)
        if not isinstance(fd_force_fp32, bool):
            raise TypeError("algorithm_kwargs['fd_force_fp32'] must be a boolean")
        self.fd_force_fp32: bool = fd_force_fp32

        # epoch progress — updated by Trainer via on_epoch_end()
        self._epoch:        int = 0
        self._total_epochs: int = 100

    # ------------------------------------------------------------------
    # Epoch hook — called by Trainer after every epoch
    # ------------------------------------------------------------------

    def on_epoch_end(self, epoch: int, total_epochs: int) -> None:
        self._epoch = epoch
        self._total_epochs = total_epochs

    @property
    def jvp_delta(self) -> float:
        """Linear anneal: jvp_delta_start → jvp_delta_end over training."""
        progress = self._epoch / max(self._total_epochs, 1)
        return self.jvp_delta_start + (self.jvp_delta_end - self.jvp_delta_start) * progress

    # ------------------------------------------------------------------
    # Module list
    # ------------------------------------------------------------------

    def trainable_modules(self) -> List[nn.Module]:
        return [self.model, self.r_embed]

    # ------------------------------------------------------------------
    # Internal forward
    # ------------------------------------------------------------------

    def _forward(self, z: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.model(self.r_embed(z, r), t)

    def _jvp_exact(
        self,
        z_t: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Return only the detached fp32 total derivative along ``(v, 0, 1)``."""
        z_f = z_t.float()
        r_f = r.float()
        t_f = t.float()
        v_f = v.float()
        with torch.no_grad():
            _, dudt = torch_jvp(
                lambda z, rv, tv: self._forward(z, rv, tv),
                (z_f, r_f, t_f),
                (v_f, torch.zeros_like(r_f), torch.ones_like(t_f)),
            )
        return dudt

    def _finite_difference_step(
        self, r: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Choose one stable, per-sample one-sided step for both z and t.

        Forward differences are used away from ``t=1``. Near the upper boundary,
        a backward step is used when it preserves ``r <= t``. For the degenerate
        diagonal point ``r=t=1`` there is no ordering-preserving non-zero step;
        a backward step is still safe because ``(t-r)=0`` removes the derivative
        from that sample's target.
        """
        delta = torch.full_like(t, self.jvp_delta)
        forward_capacity = (1.0 - t).clamp_min(0.0)
        backward_capacity = (t - r).clamp_min(0.0)
        forward = torch.minimum(delta, forward_capacity)
        backward = -torch.minimum(delta, backward_capacity)
        threshold = torch.maximum(
            delta * 0.1,
            torch.full_like(delta, torch.finfo(t.dtype).eps * 16),
        )
        can_forward = forward >= threshold
        can_backward = (-backward) >= threshold
        fallback = -torch.minimum(delta, t.clamp_min(delta))
        h = torch.where(can_forward, forward, torch.where(can_backward, backward, fallback))
        if not torch.isfinite(h).all() or (h == 0).any():
            raise FloatingPointError("Could not construct a finite non-zero FD step")
        return h

    def _finite_difference(
        self,
        z_t: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return trainable prediction, detached derivative, and consistent h."""
        h = self._finite_difference_step(r, t)
        h_image = h.view(-1, 1, 1, 1)
        z_pert = z_t + h_image * v
        t_pert = t + h
        u_pred = self._forward(z_t, r, t)
        with torch.no_grad():
            u_pred_pert = self._forward(z_pert, r, t_pert)
            dudt = (u_pred_pert - u_pred.detach()) / h_image
        return u_pred, dudt, h

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        x_data = batch
        B, dev = x_data.shape[0], x_data.device

        # Step 1 — noise + interpolant + velocity
        epsilon = torch.randn_like(x_data)
        t       = torch.rand(B, device=dev)
        v       = epsilon - x_data
        z_t     = (1.0 - t.view(-1, 1, 1, 1)) * x_data + t.view(-1, 1, 1, 1) * epsilon

        # Step 2 — sample r with optional p_same diagonal forcing
        u_uniform    = torch.rand(B, device=dev) * t
        diag_mask    = torch.rand(B, device=dev) < self.p_same
        r            = torch.where(diag_mask, t, u_uniform)

        # Step 3 — stochastic path routing
        if torch.rand(()).item() < self.p_fd_step:
            if self.use_exact_jvp:
                # The prediction owns the reverse-mode graph. The exact JVP is
                # computed separately only for the detached target, avoiding a
                # reverse-over-forward graph through torch.func.jvp.
                with torch.autocast(device_type=dev.type, enabled=False):
                    z_f, r_f, t_f, v_f = z_t.float(), r.float(), t.float(), v.float()
                    u_pred = self._forward(z_f, r_f, t_f)
                    dudt = self._jvp_exact(z_f, r_f, t_f, v_f)
                    dt = (t_f - r_f).view(-1, 1, 1, 1)
                    u_tgt = (v_f - dt * dudt).detach()
                    loss = F.mse_loss(u_pred, u_tgt)
                return {"loss": loss}

            if self.fd_force_fp32:
                with torch.autocast(device_type=dev.type, enabled=False):
                    z_f, r_f, t_f, v_f = z_t.float(), r.float(), t.float(), v.float()
                    u_pred, dudt, _ = self._finite_difference(z_f, r_f, t_f, v_f)
                    dt = (t_f - r_f).view(-1, 1, 1, 1)
                    u_tgt = (v_f - dt * dudt).detach()
                    loss = F.mse_loss(u_pred, u_tgt)
                return {"loss": loss}

            # Historical AMP behavior remains the default, but the same h is
            # now used for z, t, and the divisor, including at t=1.
            u_pred, dudt, _ = self._finite_difference(z_t, r, t, v)
            dt = (t - r).view(-1, 1, 1, 1)
            u_tgt = (v - dt * dudt).detach()
        else:
            # --- Cheap exact diagonal path ---
            # Override r = t so (t - r) == 0 → u_tgt = v exactly, no approx
            r      = t.clone()
            u_pred = self._forward(z_t, r, t)
            u_tgt  = v.detach()

        loss = F.mse_loss(u_pred, u_tgt)
        return {"loss": loss}

    # ------------------------------------------------------------------
    # Sampling — displacement identity, unchanged from original
    # ------------------------------------------------------------------

    def sample(self, n_samples: int, nfe: int, device: torch.device) -> torch.Tensor:
        C   = self.model.cfg.in_channels
        H = W = self.model._expected_image_size
        z   = torch.randn(n_samples, C, H, W, device=device)
        times = torch.linspace(1.0, 0.0, nfe + 1, device=device)

        with torch.no_grad():
            for i in range(nfe):
                t_cur  = times[i].item()
                t_next = times[i + 1].item()
                t_b = torch.full((n_samples,), t_cur,  device=device, dtype=torch.float32)
                r_b = torch.full((n_samples,), t_next, device=device, dtype=torch.float32)
                u   = self._forward(z, r_b, t_b)
                z   = z - (t_cur - t_next) * u

        return self._finalize_sample(z)
