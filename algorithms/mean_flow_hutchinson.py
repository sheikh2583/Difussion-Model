"""Mean Flow with an unbiased randomized VJP estimator of its target JVP.

For ``u(z, r, t)`` and trajectory direction ``(v, 0, 1)``, Mean Flow needs
``d = (du/dz) v + du/dt``. Given a Rademacher output probe ``xi``, reverse-mode
AD computes ``g_z = (du/dz).T xi`` and ``g_t = (du/dt).T xi``. Therefore
``d_hat = xi * (dot(g_z, v) + g_t)`` is unbiased because ``E[xi xi.T] = I``.
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
                "pixel-space configurations are not supported"
            )
        self.r_cond = RCond(model.cfg.time_embed_dim)
        self.p_same = float(self.algorithm_kwargs.get("p_same", 0.1))
        self.p_hutchinson_step = float(
            self.algorithm_kwargs.get("p_hutchinson_step", 0.8)
        )
        self.n_probes = int(self.algorithm_kwargs.get("n_probes", 1))

        if not 0.0 <= self.p_same <= 1.0:
            raise ValueError("p_same must be in [0, 1]")
        if not 0.0 <= self.p_hutchinson_step <= 1.0:
            raise ValueError("p_hutchinson_step must be in [0, 1]")
        if self.n_probes < 1:
            raise ValueError("n_probes must be >= 1")

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
        """Estimate ``(du/dz) v + du/dt`` as a detached output-shaped tensor."""
        z_leaf = z_t.detach().float().requires_grad_(True)
        t_leaf = t.detach().float().requires_grad_(True)
        r_fixed = r.detach().float()
        v_fixed = v.detach().float()

        with torch.enable_grad(), torch.autocast(
            device_type=z_t.device.type, enabled=False
        ):
            u = self._forward(z_leaf, r_fixed, t_leaf)
            estimate = torch.zeros_like(u)

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
                estimate.add_(xi * q.view(-1, 1, 1, 1))

        return (estimate / self.n_probes).detach()

    def training_step(self, batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        x_data = batch
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
