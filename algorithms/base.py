"""
Narrow algorithm interface. The Trainer and Sampler depend ONLY on
this abstract class. All algorithm-specific mathematics must live
inside concrete subclasses (flow_matching.py, mean_flow.py, mock.py)
and nowhere else.

Deliberately narrow: only `training_step` and `sample` are exposed.
There is no `prepare_batch` hook — if an algorithm needs to transform
a batch before computing its loss, that transformation happens inside
`training_step` itself, not in shared trainer code.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


class BaseAlgorithm(ABC):
    """
    Wraps a shared backbone `model` and exposes exactly two operations
    to the generic pipeline. `algorithm_kwargs` holds only parameters
    inherently required by this algorithm (never shared experimental
    controls — those are enforced upstream in ExperimentConfig / the
    runner).
    """

    def __init__(self, model: nn.Module, algorithm_kwargs: Dict[str, Any] = None):
        self.model = model
        self.algorithm_kwargs = algorithm_kwargs or {}

    @abstractmethod
    def training_step(self, batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute a loss (and optionally auxiliary scalars) for one batch
        of real images. Must return a dict containing at least the key
        "loss" (a scalar tensor with grad). Any preprocessing specific
        to this algorithm (noise sampling, interpolation, etc.) happens
        here, entirely internally.
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self, n_samples: int, nfe: int, device: torch.device) -> torch.Tensor:
        """
        Generate `n_samples` images using `nfe` function evaluations.
        Must return a tensor of shape (n_samples, C, H, W) in the training
        representation: pixel presets use [-1, 1], while latent presets use
        finite, unbounded normalized codec coordinates.

        NFE is an experimental measurement, not a guarantee of
        equivalent internal work across algorithms.
        """
        raise NotImplementedError

    def trainable_modules(self) -> List[nn.Module]:
        """
        All nn.Module components with learnable parameters that this
        algorithm owns and that the shared pipeline must optimize,
        move to the training device, switch between train/eval mode,
        and checkpoint. Defaults to just the shared backbone.

        Override ONLY if the algorithm owns additional parametric
        components beyond the shared backbone (e.g. Mean Flow's small
        r-conditioning embedding). Such components are inherent to the
        algorithm — analogous to algorithm_kwargs — and must never be
        used to alter the SHARED backbone itself, so the primary
        parameter-count/architecture comparison between algorithms
        stays apples-to-apples. Put the shared backbone first in the
        returned list by convention.
        """
        return [self.model]

    def on_after_optimizer_step(self) -> None:
        """Optional hook for state updated after every optimizer step."""

    def checkpoint_state(self) -> Dict[str, Any]:
        """Return algorithm-owned non-optimizer state for checkpointing."""
        return {}

    def load_checkpoint_state(self, state: Optional[Dict[str, Any]]) -> None:
        """Restore algorithm-owned state; defaults to no additional state."""

    def _finalize_sample(self, sample: torch.Tensor) -> torch.Tensor:
        """Apply the representation-specific output policy.

        Existing backbones do not carry ``sample_clamp`` and therefore retain
        the historical pixel-space clamp. Latent presets explicitly disable
        it, in which case non-finite output is rejected rather than silently
        corrupting a cache, decoder call, or metric.
        """
        if getattr(self.model.cfg, "sample_clamp", True):
            return sample.clamp(-1.0, 1.0)
        if not torch.isfinite(sample).all():
            raise FloatingPointError("Generated latent sample contains NaN or Inf values")
        return sample

    def name(self) -> str:
        return type(self).__name__
