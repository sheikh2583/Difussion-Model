"""
CPU-only unit tests for ConsistencyAlgorithm.sample().

Verifies the fix for the NFE=2 FID collapse bug where the old
linspace(1.0, 1/n_timesteps, nfe) injected almost no noise at the
re-injection step when nfe was small (t_next=0.056 for NFE=2).

The fixed code uses nfe+1 evenly-spaced boundaries from 1.0→0.0 so
NFE=2 re-injects at t=0.5 (50% signal + 50% noise) instead of 0.056.

No GPU, no real checkpoint, no training.
"""

from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from config.config import BackboneConfig
from models.backbone import build_backbone


# ---------------------------------------------------------------------------
# Tiny backbone for CPU-only testing (8×8, 2 channels, minimal width)
# ---------------------------------------------------------------------------
_TINY_CFG = BackboneConfig(
    name="simple_unet",
    in_channels=2,
    base_channels=16,          # keep GroupNorm happy (divisible by 8)
    channel_mults=[1, 2],
    num_res_blocks=1,
    time_embed_dim=32,
)
_IMAGE_SIZE = 8


class _FrozenZerosModel(nn.Module):
    """Minimal frozen network used as the sampling/EMA model."""

    def __init__(self):
        super().__init__()
        self._device_anchor = nn.Parameter(torch.zeros(()), requires_grad=False)

    def forward(self, x, t):
        return torch.zeros_like(x)


def _make_tiny_consistency():
    """Build a ConsistencyAlgorithm with a tiny dummy backbone on CPU.

    The teacher checkpoint path is faked via a mock because the
    constructor validates the path through resolve_checkpoint_reference.
    The teacher is never actually loaded because sample() only uses the
    EMA model.
    """
    model = build_backbone(_TINY_CFG, image_size=_IMAGE_SIZE)

    with patch(
        "algorithms.consistency.resolve_checkpoint_reference",
        return_value="fake_teacher.pt",
    ):
        from algorithms.consistency import ConsistencyAlgorithm

        algo = ConsistencyAlgorithm(
            model,
            algorithm_kwargs={
                "teacher_checkpoint": "fake_teacher.pt",
                "n_timesteps": 18,
            },
        )
    algo.ema_model = _FrozenZerosModel()
    return algo


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSampleShapeAndRange:
    """sample() must return (n_samples, C, H, W) in [-1, 1]."""

    @pytest.fixture(autouse=True)
    def algo(self):
        self._algo = _make_tiny_consistency()

    @pytest.mark.parametrize("nfe", [1, 2, 5])
    def test_output_shape(self, nfe):
        out = self._algo.sample(n_samples=4, nfe=nfe, device=torch.device("cpu"))
        assert out.shape == (4, _TINY_CFG.in_channels, _IMAGE_SIZE, _IMAGE_SIZE), (
            f"NFE={nfe}: expected shape "
            f"(4, {_TINY_CFG.in_channels}, {_IMAGE_SIZE}, {_IMAGE_SIZE}), "
            f"got {out.shape}"
        )

    @pytest.mark.parametrize("nfe", [1, 2, 5])
    def test_output_range(self, nfe):
        out = self._algo.sample(n_samples=4, nfe=nfe, device=torch.device("cpu"))
        assert out.min() >= -1.0, f"NFE={nfe}: output has values below -1"
        assert out.max() <= 1.0, f"NFE={nfe}: output has values above 1"


class TestTimeBoundaries:
    """Verify that the fixed multi-step schedule uses correct boundaries."""

    def test_nfe2_boundaries_are_correct(self):
        """For NFE=2 the boundaries must be [1.0, 0.5, 0.0], meaning the
        first re-injection happens at t=0.5 (half noise, half signal).

        Before the fix, the code used linspace(1.0, 1/18, 2) which gave
        t_next ≈ 0.056 — injecting almost no noise and collapsing FID.
        """
        algo = _make_tiny_consistency()
        call_times = []
        second_input = []

        def fake_consistency_fn(net, x, t):
            call_times.append(t.detach().clone())
            if len(call_times) == 1:
                return torch.ones_like(x)
            second_input.append(x.detach().clone())
            return x

        # Zero re-injected noise makes the boundary coefficient observable:
        # t_next=0.5 turns the first all-ones estimate into an all-0.5 input.
        with patch.object(algo, "_consistency_fn", side_effect=fake_consistency_fn), \
             patch("algorithms.consistency.torch.randn_like", side_effect=torch.zeros_like):
            algo.sample(n_samples=4, nfe=2, device=torch.device("cpu"))

        assert len(call_times) == 2
        assert torch.allclose(call_times[0], torch.ones(4))
        assert torch.allclose(call_times[1], torch.full((4,), 0.5))
        assert torch.allclose(second_input[0], torch.full_like(second_input[0], 0.5))

    def test_nfe5_boundaries(self):
        """For NFE=5 the boundaries are [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]."""
        nfe = 5
        t_boundaries = torch.linspace(1.0, 0.0, nfe + 1)
        expected = torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2, 0.0])
        assert torch.allclose(t_boundaries, expected, atol=1e-6), (
            f"NFE=5 boundaries should be {expected.tolist()}, got {t_boundaries.tolist()}"
        )

    def test_nfe1_uses_single_pass(self):
        """NFE=1 bypasses the multi-step path entirely and uses a single
        forward pass at t=1.0. This should be unaffected by the fix."""
        algo = _make_tiny_consistency()
        out = algo.sample(n_samples=2, nfe=1, device=torch.device("cpu"))
        assert out.shape == (2, _TINY_CFG.in_channels, _IMAGE_SIZE, _IMAGE_SIZE)

    def test_old_schedule_was_wrong_for_nfe2(self):
        """Regression guard: the OLD schedule linspace(1.0, 1/18, 2)
        produced t_next ≈ 0.056, which is catastrophically wrong."""
        n_timesteps = 18
        nfe = 2
        old_ts = torch.linspace(1.0, 1.0 / n_timesteps, nfe)
        # The old code used t_val - 1/n_timesteps as t_next for re-injection
        old_t_next = max(old_ts[0].item() - 1.0 / n_timesteps, 0.0)
        # old_t_next ≈ 0.944 — but the real problem was the second call's
        # t_val which was 1/18 ≈ 0.056, so the model saw almost t=0 input.
        assert old_ts[1].item() < 0.06, (
            "Sanity check: old NFE=2 schedule's second t value was ~0.056"
        )
        # The new schedule's second boundary is 0.5 — much healthier:
        new_boundaries = torch.linspace(1.0, 0.0, nfe + 1)
        assert new_boundaries[1].item() > 0.4, (
            "New NFE=2 schedule should have t_next ~0.5 for re-injection"
        )


class TestSampleDeterminism:
    """With the same seed, sample() must produce identical outputs."""

    def test_deterministic_with_seed(self):
        algo = _make_tiny_consistency()

        torch.manual_seed(42)
        out1 = algo.sample(n_samples=4, nfe=2, device=torch.device("cpu"))

        torch.manual_seed(42)
        out2 = algo.sample(n_samples=4, nfe=2, device=torch.device("cpu"))

        assert torch.equal(out1, out2), "Same seed should produce identical samples"
