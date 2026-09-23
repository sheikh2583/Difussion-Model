"""Mean Flow with a control-variate randomized VJP target estimator.

For ``u(z, r, t)`` and trajectory direction ``(v, 0, 1)``, Mean Flow needs
``d = (du/dz) v + du/dt``. Given a Rademacher output probe ``xi``, reverse-mode
AD computes ``q = xi.T d``.  A finite-difference baseline ``d_fd`` removes
most of the high-dimensional probe variance while preserving unbiasedness:

``d_hat = d_fd + xi * (q - xi.T d_fd)``.

Because ``E[xi xi.T] = I``, ``E[d_hat] = d``. Its variance depends on the
finite-difference residual ``d - d_fd`` instead of the full derivative ``d``.
"""

from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.base import BaseAlgorithm
from algorithms.r_embed import RCond


class MeanFlowHutchinsonAlgorithm(BaseAlgorithm):
    """Latent-space diagnostic Mean Flow variant using randomized VJP probes."""

    def __init__(self, model: nn.Module, algorithm_kwargs: Dict[str, Any] = None):
        super().__init__(model, algorithm_kwargs)
        if getattr(model.cfg, "sample_clamp", True):
            raise ValueError(
                "mf_hutchinson is restricted to latent-space backbones; "
                "pixel-space models are not supported"
            )
        self.r_cond = RCond(model.cfg.time_embed_dim)
        self.p_same = float(self.algorithm_kwargs.get("p_same", 0.1))
        self.p_hutchinson_step = float(
            self.algorithm_kwargs.get("p_hutchinson_step", 0.8)
        )
        self.n_probes = int(self.algorithm_kwargs.get("n_probes", 1))
        self.fd_eps_start = float(
            self.algorithm_kwargs.get("fd_eps_start", 1e-2)
        )
        self.fd_eps_end = float(
            self.algorithm_kwargs.get("fd_eps_end", 1e-4)
        )
        self._epoch = 0
        self._total_epochs = 100

        if not 0.0 <= self.p_same <= 1.0:
            raise ValueError("p_same must be in [0, 1]")
        if not 0.0 <= self.p_hutchinson_step <= 1.0:
            raise ValueError("p_hutchinson_step must be in [0, 1]")
        if self.n_probes < 1:
            raise ValueError("n_probes must be >= 1")
        if self.fd_eps_start <= 0 or self.fd_eps_end <= 0:
            raise ValueError("fd_eps_start and fd_eps_end must be positive")

    def on_epoch_end(self, epoch: int, total_epochs: int) -> None:
        """Advance the finite-difference control-variate schedule."""
        self._epoch = epoch
        self._total_epochs = total_epochs

    @property
    def fd_eps(self) -> float:
        """Linearly anneal the control-variate step over training."""
        progress = self._epoch / max(self._total_epochs, 1)
        return self.fd_eps_start + (
            self.fd_eps_end - self.fd_eps_start
        ) * progress

    def trainable_modules(self) -> List[nn.Module]:
        return [self.model, self.r_cond]

    def _forward(
        self, z: torch.Tensor, r: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        t_emb = self.model.time_embed(t)
        r_emb = self.r_cond(r)
        return self._backbone_forward(z, t_emb + r_emb)

    def _backbone_forward(
        self, z: torch.Tensor, t_emb: torch.Tensor
    ) -> torch.Tensor:
        """Run SimpleUNet with an externally supplied time embedding."""
        m = self.model
        h = m.in_conv(z)
        skips = [h]
        for stage, down in zip(m.down_blocks, m.downsamples):
            for block in stage:
                h = block(h, t_emb)
                skips.append(h)
            h = down(h)
            if not isinstance(down, nn.Identity):
                skips.append(h)
        h = m.mid1(h, t_emb)
        h = m.mid2(h, t_emb)
        for stage, up in zip(m.up_blocks, m.upsamples):
            for block in stage:
                skip = skips.pop()
                h = block(torch.cat([h, skip], dim=1), t_emb)
            h = up(h)
        return m.out_conv(F.silu(m.out_norm(h)))

    def _hutchinson_dudt(
        self,
        z_t: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate ``(du/dz) v + du/dt`` with an FD control variate.

        For each Rademacher probe ``xi``, reverse-mode AD supplies
        ``q = xi.T d``.  The unbiased correction estimates only the residual
        between ``d`` and the finite-difference baseline ``d_fd``:

        ``d_hat = d_fd + mean[xi * (q - xi.T d_fd)]``.
        """
        batch_size = z_t.shape[0]
        z_leaf = z_t.detach().float().requires_grad_(True)
        t_leaf = t.detach().float().requires_grad_(True)
        r_fixed = r.detach().float()
        v_fixed = v.detach().float()

        with torch.enable_grad(), torch.autocast(
            device_type=z_t.device.type, enabled=False
        ):
            u = self._forward(z_leaf, r_fixed, t_leaf)

            # Use the same boundary-safe, per-sample step for z, t, and the
            # divisor. Forward differences are preferred; near t=1 a backward
            # step avoids evaluating the time embedding outside [0, 1].
            delta = torch.full_like(t_leaf, self.fd_eps)
            forward_capacity = (1.0 - t_leaf).clamp_min(0.0)
            backward_capacity = (t_leaf - r_fixed).clamp_min(0.0)
            forward = torch.minimum(delta, forward_capacity)
            backward = -torch.minimum(delta, backward_capacity)
            threshold = torch.maximum(
                delta * 0.1,
                torch.full_like(delta, torch.finfo(t_leaf.dtype).eps * 16),
            )
            can_forward = forward >= threshold
            can_backward = (-backward) >= threshold
            fallback = -torch.minimum(delta, t_leaf.clamp_min(delta))
            step = torch.where(
                can_forward,
                forward,
                torch.where(can_backward, backward, fallback),
            )
            if not torch.isfinite(step).all() or (step == 0).any():
                raise FloatingPointError(
                    "Could not construct a finite non-zero Hutchinson FD step"
                )
            step_image = step.view(-1, 1, 1, 1)

            with torch.no_grad():
                u_perturbed = self._forward(
                    z_leaf.detach() + step_image * v_fixed,
                    r_fixed,
                    t_leaf.detach() + step,
                )
            d_fd = (u_perturbed - u.detach()) / step_image

            correction = torch.zeros_like(u)

            for probe_index in range(self.n_probes):
                xi = torch.empty_like(u).bernoulli_(0.5).mul_(2).sub_(1)
                scalar = (xi * u).sum()
                g_z, g_t = torch.autograd.grad(
                    scalar,
                    (z_leaf, t_leaf),
                    retain_graph=probe_index + 1 < self.n_probes,
                    create_graph=False,
                )
                q = (g_z * v_fixed).flatten(1).sum(1) + g_t
                q_fd = (xi * d_fd).flatten(1).sum(1)
                correction.add_(
                    xi * (q - q_fd).view(batch_size, 1, 1, 1)
                )

        return (d_fd + correction / self.n_probes).detach()

    def training_step(self, batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        x_data = batch

        # Latent-only guard: the CV estimator is tuned for VQ-f4 latents.
        if x_data.ndim != 4 or x_data.shape[-2:] != (16, 16):
            spatial_size = tuple(x_data.shape[-2:]) if x_data.ndim >= 2 else ()
            raise ValueError(
                "MeanFlowHutchinsonAlgorithm expects latent inputs (H=W=16), "
                f"got spatial size {spatial_size}. "
                "Use MeanFlowAlgorithm for pixel-space training."
            )

        batch_size, device = x_data.shape[0], x_data.device

        epsilon = torch.randn_like(x_data)
        t = torch.rand(batch_size, device=device)
        v = epsilon - x_data
        t_image = t.view(-1, 1, 1, 1)
        z_t = (1.0 - t_image) * x_data + t_image * epsilon

        r_uniform = torch.rand(batch_size, device=device) * t
        diagonal = torch.rand(batch_size, device=device) < self.p_same
        r = torch.where(diagonal, t, r_uniform)

        if torch.rand((), device=device).item() < self.p_hutchinson_step:
            dudt = self._hutchinson_dudt(z_t, r, t, v)
            dt = (t - r).view(-1, 1, 1, 1)
            u_target = (v.float() - dt.float() * dudt).detach()
            u_prediction = self._forward(z_t, r, t)
        else:
            r = t.clone()
            u_prediction = self._forward(z_t, r, t)
            u_target = v.detach()

        loss = F.mse_loss(u_prediction.float(), u_target.float())
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite MF-Hutchinson loss: {loss.item()}")
        return {"loss": loss}

    def sample(
        self, n_samples: int, nfe: int, device: torch.device
    ) -> torch.Tensor:
        channels = self.model.cfg.in_channels
        height = width = self.model._expected_image_size
        z = torch.randn(n_samples, channels, height, width, device=device)
        times = torch.linspace(1.0, 0.0, nfe + 1, device=device)

        with torch.no_grad():
            for index in range(nfe):
                t_current = times[index].item()
                t_next = times[index + 1].item()
                t_batch = torch.full((n_samples,), t_current, device=device)
                r_batch = torch.full((n_samples,), t_next, device=device)
                velocity = self._forward(z, r_batch, t_batch)
                z = z - (t_current - t_next) * velocity

        return self._finalize_sample(z)
