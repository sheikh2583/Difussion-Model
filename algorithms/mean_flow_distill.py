"""
Mean Flow Distillation — 4th algorithm (MF-Distill).

Trains a Mean Flow student without any JVP or finite-difference by using
a frozen Flow Matching teacher's multi-step Euler rollout as the target.

How it differs from MeanFlowAlgorithm
--------------------------------------
MeanFlowAlgorithm computes u_tgt analytically via the Mean Flow Identity
(exact JVP or FD approximation).  This algorithm replaces that with:

    z_r_teacher = frozen_FM_teacher.euler(z_t, t → r, teacher_nfe steps)
    u_tgt       = (z_t - z_r_teacher) / (t - r)

This is a self-distillation target: the empirical average velocity
measured by running the teacher, not a derivative of the student.

Architectural fairness note
----------------------------
The student backbone is identical in parameter count to FM / MF (built
from the same BackboneConfig).  The teacher is an extra cost that sits
outside the fair comparison, and must be reported separately in results.
See algorithms/__init__.py and results.py for labelling conventions.

algorithm_kwargs
----------------
    teacher_checkpoint (str, required):
        Path to a trained FlowMatchingAlgorithm checkpoint, e.g.
        "results/fm_cifar10/checkpoints/FlowMatchingAlgorithm_epoch100.pt"
    teacher_nfe (int, 4):
        Euler steps for teacher rollout per training step.
    p_same (float, 0.25):
        Probability of diagonal (r == t) shortcut per sample.
"""

from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.base import BaseAlgorithm
from algorithms.r_embed import REmbed
from models.backbone import build_backbone


class MeanFlowDistillAlgorithm(BaseAlgorithm):

    def __init__(self, model: nn.Module, algorithm_kwargs: Dict[str, Any] = None):
        super().__init__(model, algorithm_kwargs)

        # --- student r-embed (same as MeanFlowAlgorithm, via shared module) ---
        C = model.cfg.in_channels
        self.r_embed = REmbed(C)

        # --- hyper-parameters ---
        self.p_same:      float = float(self.algorithm_kwargs.get("p_same", 0.25))
        self.teacher_nfe: int   = int(self.algorithm_kwargs.get("teacher_nfe", 4))

        # --- frozen teacher ---
        ckpt_path = self.algorithm_kwargs.get("teacher_checkpoint")
        if not ckpt_path:
            raise ValueError(
                "MeanFlowDistillAlgorithm requires algorithm_kwargs['teacher_checkpoint'] "
                "pointing to a trained FlowMatchingAlgorithm checkpoint."
            )
        self.teacher = build_backbone(model.cfg, image_size=model._expected_image_size)
        state = torch.load(ckpt_path, map_location="cpu")
        # checkpoint may be a full trainer dict or a raw state dict
        sd = state.get("model_state", state)
        self.teacher.load_state_dict(sd)
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()

    # ------------------------------------------------------------------
    # Module list — teacher excluded from optimizer by design
    # ------------------------------------------------------------------

    def trainable_modules(self) -> List[nn.Module]:
        # teacher is deliberately omitted — frozen, never optimized
        return [self.model, self.r_embed]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _student_forward(self, z: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.model(self.r_embed(z, r), t)

    # Alias so AdaptiveMeanFlowSampler (and any future wrapper that calls ._forward)
    # works with MeanFlowDistillAlgorithm without modification.
    _forward = _student_forward


    @torch.no_grad()
    def _teacher_velocity(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Single frozen-teacher forward pass (FM: plain backbone, no r)."""
        return self.teacher(z, t)

    @torch.no_grad()
    def _teacher_rollout(
        self,
        z_t: torch.Tensor,   # (B, C, H, W) at time t
        t:   torch.Tensor,   # (B,) start times
        r:   torch.Tensor,   # (B,) end times (r <= t)
    ) -> torch.Tensor:
        """
        Euler-integrate the frozen teacher from t down to r in
        self.teacher_nfe steps, returning z at time r.
        Runs under torch.no_grad() — all inference, no grad tape.
        """
        z = z_t.clone()
        B = z.shape[0]
        nfe = self.teacher_nfe
        for i in range(nfe):
            alpha     = i / nfe
            alpha_nxt = (i + 1) / nfe
            t_cur  = t - (t - r) * alpha        # (B,) between t and r
            t_next = t - (t - r) * alpha_nxt
            v = self.teacher(z, t_cur)
            z = z - (t_cur - t_next).view(-1, 1, 1, 1) * v
        return z  # z at time r

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

        # Move teacher to same device lazily
        if next(self.teacher.parameters()).device != dev:
            self.teacher.to(dev)

        # Step 2 — sample r with optional diagonal forcing
        u_uniform = torch.rand(B, device=dev) * t
        diag_mask = torch.rand(B, device=dev) < self.p_same
        r         = torch.where(diag_mask, t, u_uniform)

        # Step 3 — build distillation target (no derivatives needed)
        dt       = (t - r)                                    # (B,)
        same     = dt < 1e-6                                  # bool (B,)
        u_tgt    = torch.zeros_like(z_t)

        # Diagonal: u_tgt = instantaneous teacher velocity (single call)
        if same.any():
            idx         = same.nonzero(as_tuple=True)[0]
            u_tgt[idx]  = self._teacher_velocity(z_t[idx], t[idx])

        # Interval: u_tgt = (z_t - z_r) / (t - r) via teacher Euler rollout
        if (~same).any():
            idx        = (~same).nonzero(as_tuple=True)[0]
            z_r        = self._teacher_rollout(z_t[idx], t[idx], r[idx])
            u_tgt[idx] = (z_t[idx] - z_r) / dt[idx].view(-1, 1, 1, 1).clamp(min=1e-6)

        u_tgt  = u_tgt.detach()
        u_pred = self._student_forward(z_t, r, t)
        loss   = F.mse_loss(u_pred, u_tgt)
        return {"loss": loss}

    # ------------------------------------------------------------------
    # Sampling — student uses displacement identity, same as MF
    # ------------------------------------------------------------------

    def sample(self, n_samples: int, nfe: int, device: torch.device) -> torch.Tensor:
        C   = self.model.cfg.in_channels
        H = W = self.model._expected_image_size
        z   = torch.randn(n_samples, C, H, W, device=device)
        times = torch.linspace(1.0, 0.0, nfe + 1, device=device)

        if next(self.teacher.parameters()).device != device:
            self.teacher.to(device)

        with torch.no_grad():
            for i in range(nfe):
                t_cur  = times[i].item()
                t_next = times[i + 1].item()
                t_b = torch.full((n_samples,), t_cur,  device=device, dtype=torch.float32)
                r_b = torch.full((n_samples,), t_next, device=device, dtype=torch.float32)
                z   = z - (t_cur - t_next) * self._student_forward(z, r_b, t_b)

        return z.clamp(-1.0, 1.0)
