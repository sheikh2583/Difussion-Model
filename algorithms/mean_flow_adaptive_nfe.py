"""
Future Direction 01: Adaptive MeanFlow for Dynamic NFE.

SLIDE REFERENCE: "Adaptive NFE Allocation" — thesis presentation, future work.

Background
----------
MeanFlow's displacement identity enables exact jumps between arbitrary time
points, making it uniquely suited to adaptive (per-sample) scheduling.
However, the current Sampler always runs a fixed NFE for every sample in a
batch, even when some samples have already converged to high-confidence
predictions after just one or two steps.

Observation: generation difficulty is uneven across CIFAR-10.  Uniform
backgrounds (sky, solid-colour regions) converge in 1 step; fine textures
(fur, vehicle grilles, foliage) require more.  A fixed NFE wastes compute on
easy samples and under-allocates on hard ones.

Planned Approach
----------------
1. Start with all N samples active (active_mask: BoolTensor of shape (B,)).
2. At each step i in range(max_nfe):
   a. Run _forward only on active samples: u = algorithm._forward(z[active], r[active], t[active])
   b. Update z for active samples: z[active] -= step_size * u
   c. Compute per-sample confidence:
          conf = ||u - u_prev||_2 / (||u||_2 + eps)    [shape (B_active,)]
      where u_prev is the velocity from the previous step (zero on first step).
   d. Deactivate samples where conf < confidence_threshold AND i >= min_nfe.
   e. Break early if active_mask.sum() == 0.
3. Return z.clamp(-1, 1).

Why this matters
----------------
- Preserves MeanFlow's low *average* NFE while recovering quality on hard cases.
- No retraining needed: works on any MeanFlowAlgorithm checkpoint.
- Expected outcome: FID improves over 1-step baseline at ~1.3–1.8 average NFE.

Prerequisites before implementing
----------------------------------
- Decide time schedule: uniform linspace vs. learned schedule.
- Decide how to handle the batch-dimension mismatch when only a subset is active
  (masked indexing vs. padding vs. dynamic batch shrink).
- Profile memory vs. dynamic-batch approach on RTX 3060 (12 GB).

NOT a BaseAlgorithm subclass.  This is a pure sampling wrapper;
it does not define a training_step.  The wrapped algorithm must expose
._forward(z, r, t) (as MeanFlowAlgorithm does).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from algorithms.mean_flow import MeanFlowAlgorithm


class AdaptiveMeanFlowSampler:
    """
    Sampling wrapper that allocates NFE per sample based on estimated
    generation difficulty, stopping early when a sample's velocity
    estimate has stabilised.

    Parameters
    ----------
    algorithm : MeanFlowAlgorithm
        A fully trained student (or standard) MeanFlow algorithm.
        Must expose ._forward(z, r, t).
    min_nfe : int
        Minimum number of displacement steps every sample receives
        before it can be deactivated.  Default: 1.
    max_nfe : int
        Hard upper bound on steps.  No sample exceeds this.  Default: 4.
    confidence_threshold : float
        Fractional change in velocity below which a sample is considered
        converged:  ||u_t - u_{t-1}|| / ||u_t|| < threshold.  Default: 0.05.
    """

    def __init__(
        self,
        algorithm: "MeanFlowAlgorithm",
        min_nfe: int = 1,
        max_nfe: int = 4,
        confidence_threshold: float = 0.05,
    ):
        self.algorithm            = algorithm
        self.min_nfe              = min_nfe
        self.max_nfe              = max_nfe
        self.confidence_threshold = confidence_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample(self, n_samples: int, device: torch.device) -> torch.Tensor:
        """
        Generate `n_samples` images with adaptive NFE allocation.

        Returns
        -------
        torch.Tensor
            Shape (n_samples, C, H, W), values clamped to [-1, 1].
        """
        raise NotImplementedError(
            "TODO: implement per-sample early exit.\n"
            "\n"
            "Outline:\n"
            "  C = self.algorithm.model.cfg.in_channels\n"
            "  H = W = self.algorithm.model._expected_image_size\n"
            "  z = torch.randn(n_samples, C, H, W, device=device)     # start from noise\n"
            "  times = torch.linspace(1.0, 0.0, self.max_nfe+1, device=device)\n"
            "  active_mask = torch.ones(n_samples, dtype=torch.bool, device=device)\n"
            "  u_prev = torch.zeros_like(z)                           # velocity memory\n"
            "\n"
            "  with torch.no_grad():\n"
            "      for i in range(self.max_nfe):\n"
            "          t_val  = times[i].item()\n"
            "          r_val  = times[i + 1].item()\n"
            "          step   = t_val - r_val\n"
            "          idx    = active_mask.nonzero(as_tuple=True)[0]\n"
            "          t_b    = torch.full((idx.numel(),), t_val, device=device)\n"
            "          r_b    = torch.full((idx.numel(),), r_val, device=device)\n"
            "          u      = self.algorithm._forward(z[idx], r_b, t_b)  # active only\n"
            "          z[idx] = z[idx] - step * u\n"
            "\n"
            "          # Confidence: fractional velocity change per sample\n"
            "          diff = (u - u_prev[idx]).flatten(1).norm(dim=1)\n"
            "          mag  = u.flatten(1).norm(dim=1).clamp(min=1e-8)\n"
            "          conf = diff / mag                              # (B_active,)\n"
            "          u_prev[idx] = u\n"
            "\n"
            "          if i + 1 >= self.min_nfe:\n"
            "              converged = conf < self.confidence_threshold\n"
            "              active_mask[idx[converged]] = False\n"
            "\n"
            "          if not active_mask.any():\n"
            "              break\n"
            "\n"
            "  return z.clamp(-1.0, 1.0)"
        )
