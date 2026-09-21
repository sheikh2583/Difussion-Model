"""
Consistency Models — Song et al. 2023, arXiv:2303.01469.

See THEORY_NOTES.md §5 for full mathematical derivation.

Core idea
---------
Train a consistency function f_θ(x_t, t) → x_0 that maps ANY point on
ANY trajectory directly to the clean image at t=0.  The self-consistency
constraint (points on the same trajectory map to the same x_0) is enforced
via a regression loss between consecutive steps.

Self-consistency loss:
    L_CD = E[ ||f_θ(x_t, t) - f_θ-(x_{t-Δt}, t-Δt)||² ]
where:
    x_{t-Δt}  = x_t - Δt·v_teacher(x_t, t)   [one teacher Euler step]
    f_θ-       = EMA of f_θ (stable target)
    f_θ(x,t)  = c_skip(t)·x + c_out(t)·F_θ(x,t)   [boundary-enforcing parameterization]
    c_skip(t) = 1/(1+t²),   c_out(t) = t/√(1+t²)  [ensures f_θ(x,0)=x]

algorithm_kwargs
----------------
    teacher_checkpoint (str, required): path to trained FM checkpoint.
    ema_decay          (float, 0.999):  EMA decay for target network.
    consistency_weight (float, 1.0):    loss scale (tune if training collapses).
    n_timesteps        (int,  18):      discretisation steps for t schedule.

Training notes (read before tuning)
-------------------------------------
- If loss explodes after epoch 3: halve consistency_weight (try 0.5, then 0.25).
- If loss stagnates (never decreases below first-epoch value): lower ema_decay
  to 0.995 so the EMA target moves faster.
- c_skip/c_out boundary condition is critical — do not remove it.
- Expect 2–3 debugging cycles; this is normal for CM training.
"""

import copy
import warnings
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.base import BaseAlgorithm
from models.backbone import build_backbone
from utils.checkpoints import extract_model_state, resolve_checkpoint_reference


class ConsistencyAlgorithm(BaseAlgorithm):

    def __init__(self, model: nn.Module, algorithm_kwargs: Dict[str, Any] = None):
        super().__init__(model, algorithm_kwargs)

        # --- hyper-parameters ---
        self.ema_decay          = float(self.algorithm_kwargs.get("ema_decay", 0.999))
        self.consistency_weight = float(self.algorithm_kwargs.get("consistency_weight", 1.0))
        self.n_timesteps        = int(self.algorithm_kwargs.get("n_timesteps", 18))

        # --- EMA target network (frozen copy of student, updated by EMA not grad) ---
        # Deep copy so it has its own parameters; they are excluded from optimizer
        # because trainable_modules() returns only [self.model].
        self.ema_model = copy.deepcopy(model)
        for p in self.ema_model.parameters():
            p.requires_grad = False
        self.ema_model.eval()

        # --- frozen FM teacher for one-step Euler targets ---
        ckpt_path = self.algorithm_kwargs.get("teacher_checkpoint")
        if not ckpt_path:
            raise ValueError(
                "ConsistencyAlgorithm requires algorithm_kwargs['teacher_checkpoint'] "
                "pointing to a trained FlowMatchingAlgorithm checkpoint."
            )
        self.teacher_checkpoint = str(resolve_checkpoint_reference(ckpt_path))
        self.teacher = None

        # Discrete time schedule: n_timesteps points in (0, 1]
        self._ts = torch.linspace(1.0 / self.n_timesteps, 1.0, self.n_timesteps)

    # ------------------------------------------------------------------
    # Module list — EMA model and teacher excluded from optimizer
    # ------------------------------------------------------------------

    def trainable_modules(self) -> List[nn.Module]:
        # Only the student model is optimized.
        # ema_model and teacher are updated/frozen separately.
        return [self.model]

    # ------------------------------------------------------------------
    # Boundary-enforcing parameterization
    # ------------------------------------------------------------------

    def _c_skip(self, t: torch.Tensor) -> torch.Tensor:
        """c_skip(t) = 1/(1+t²) — ensures f_θ(x,0)=x at t=0."""
        return 1.0 / (1.0 + t ** 2)

    def _c_out(self, t: torch.Tensor) -> torch.Tensor:
        """c_out(t) = t/√(1+t²)"""
        return t / torch.sqrt(1.0 + t ** 2)

    def _consistency_fn(self, net: nn.Module, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        f_θ(x, t) = c_skip(t)·x + c_out(t)·net(x, t)
        Enforces f_θ(x, 0) = x exactly via the c_skip/c_out schedule.
        """
        c_skip = self._c_skip(t).view(-1, 1, 1, 1)
        c_out  = self._c_out(t).view(-1, 1, 1, 1)
        return c_skip * x + c_out * net(x, t)

    def _ensure_teacher(self, device: torch.device) -> nn.Module:
        """Load the frozen teacher only when a training step actually needs it."""
        if self.teacher is None:
            teacher = build_backbone(
                self.model.cfg, image_size=self.model._expected_image_size
            )
            state = torch.load(
                self.teacher_checkpoint, map_location="cpu", weights_only=False
            )
            teacher.load_state_dict(extract_model_state(state))
            for parameter in teacher.parameters():
                parameter.requires_grad = False
            teacher.eval()
            self.teacher = teacher
        if next(self.teacher.parameters()).device != device:
            self.teacher.to(device)
        return self.teacher

    # ------------------------------------------------------------------
    # EMA update (called after each optimizer step)
    # ------------------------------------------------------------------

    def update_ema(self) -> None:
        """Update the EMA target after a student optimizer step."""
        with torch.no_grad():
            for p_ema, p_student in zip(self.ema_model.parameters(),
                                         self.model.parameters()):
                p_ema.data.mul_(self.ema_decay).add_(p_student.data, alpha=1 - self.ema_decay)

    def on_after_optimizer_step(self) -> None:
        """Hook called by Trainer after each student optimizer step."""
        self.update_ema()

    def checkpoint_state(self) -> Dict[str, Any]:
        return {"ema_model": self.ema_model.state_dict()}

    def load_checkpoint_state(self, state: Optional[Dict[str, Any]]) -> None:
        if state and "ema_model" in state:
            self.ema_model.load_state_dict(state["ema_model"])
            return
        # Legacy checkpoints did not persist EMA weights. Copying the trained
        # student is a safe fallback; retaining a random EMA would invalidate
        # sampling and evaluation.
        self.ema_model.load_state_dict(self.model.state_dict())
        warnings.warn(
            "Consistency checkpoint has no EMA state; initialized EMA from "
            "the student weights for legacy compatibility.",
            RuntimeWarning,
            stacklevel=2,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        x_data = batch
        B, dev = x_data.shape[0], x_data.device

        # Load/move teacher and EMA model lazily. Sampling never needs teacher.
        teacher = self._ensure_teacher(dev)
        if next(self.ema_model.parameters()).device != dev:
            self.ema_model.to(dev)

        # Step 1 — sample a discrete time index and build x_t
        ts = self._ts.to(dev)
        idx = torch.randint(0, self.n_timesteps, (B,), device=dev)
        t   = ts[idx]                                           # (B,)

        epsilon = torch.randn_like(x_data)
        t_view  = t.view(-1, 1, 1, 1)
        x_t     = (1.0 - t_view) * x_data + t_view * epsilon  # (B, C, H, W)

        # Step 2 — one Euler step with frozen teacher to get x_{t-Δt}
        with torch.no_grad():
            dt = 1.0 / self.n_timesteps
            v_teacher = teacher(x_t, t)                        # (B, C, H, W)
            t_prev    = (t - dt).clamp(min=0.0)
            x_t_prev  = x_t - dt * v_teacher                  # one step toward data

        # Step 3 — consistency loss: student at t vs EMA target at t-Δt
        # Student prediction at current t
        f_student = self._consistency_fn(self.model, x_t, t)

        # EMA target prediction at previous t (stop-gradient via no_grad on ema_model)
        with torch.no_grad():
            f_target = self._consistency_fn(self.ema_model, x_t_prev, t_prev)

        loss = self.consistency_weight * F.mse_loss(f_student, f_target)
        return {"loss": loss}

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, n_samples: int, nfe: int, device: torch.device) -> torch.Tensor:
        """
        One-step sampling: f_θ(x_1, 1) maps pure noise directly to image.
        Multi-step: alternately apply f_θ and re-inject noise for refinement.
        """
        C   = self.model.cfg.in_channels
        H = W = self.model._expected_image_size

        if next(self.ema_model.parameters()).device != device:
            self.ema_model.to(device)

        with torch.no_grad():
            x = torch.randn(n_samples, C, H, W, device=device)

            if nfe == 1:
                t = torch.ones(n_samples, device=device)
                x = self._consistency_fn(self.ema_model, x, t)
            else:
                # Multi-step: nfe forward passes with noise re-injection between steps.
                # Use nfe+1 evenly-spaced time boundaries from 1.0 to 0.0 so each
                # step sees a meaningful noise level. This fixes the NFE=2 collapse
                # (previous linspace(1.0, 1/n_timesteps, nfe) gave t_next=0.056 for
                # NFE=2, injecting almost no noise and producing FID=434 vs NFE=1's 137).
                t_boundaries = torch.linspace(1.0, 0.0, nfe + 1, device=device)
                for i in range(nfe):
                    t_cur  = t_boundaries[i]
                    t_next = t_boundaries[i + 1]
                    t_b    = torch.full((n_samples,), t_cur.item(), device=device)
                    x_0_hat = self._consistency_fn(self.ema_model, x, t_b)
                    if i < nfe - 1:
                        # Re-inject noise at the next time level
                        noise = torch.randn_like(x_0_hat)
                        x = (1.0 - t_next) * x_0_hat + t_next * noise
                    else:
                        x = x_0_hat

        return self._finalize_sample(x)
