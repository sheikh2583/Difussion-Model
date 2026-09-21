"""
tests/test_sampler_channel_guard.py — CPU-only tests for latent-grid safety.

Tests cover:
  - 3-channel normalized latent output: save_image is NOT called
  - 3-channel RGB output: save_image IS called (existing behaviour preserved)
  - save_grid=False: save_image is never called regardless of channels
  - Warning is logged for latent suppression
"""
import logging
import os
import tempfile
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from algorithms.base import BaseAlgorithm
from sampling.sampler import Sampler


# ── Fake algorithms ────────────────────────────────────────────────────────────
class _LatentAlg(BaseAlgorithm):
    """Returns 3-channel normalized VQ latents."""
    def training_step(self, batch): return {"loss": torch.tensor(0.0)}
    def sample(self, n, nfe, device):
        return torch.randn(n, 3, 16, 16)


class _PixelAlg(BaseAlgorithm):
    """Returns 3-channel RGB images."""
    def training_step(self, batch): return {"loss": torch.tensor(0.0)}
    def sample(self, n, nfe, device):
        return torch.zeros(n, 3, 64, 64)


class _TwoChannelAlg(BaseAlgorithm):
    """Returns 2-channel output (edge case)."""
    def training_step(self, batch): return {"loss": torch.tensor(0.0)}
    def sample(self, n, nfe, device):
        return torch.zeros(n, 2, 32, 32)


# ── Helper ────────────────────────────────────────────────────────────────────
def _run_sampler_run(alg, n_samples=2, nfe_values=(1,), save_grid=True):
    """Run Sampler.run() with save_image patched; return list of save_image calls."""
    with tempfile.TemporaryDirectory() as tmp:
        sampler = Sampler(alg, torch.device("cpu"), tmp, "test", seed=0)
        saved = []
        with mock.patch(
            "sampling.sampler.save_image",
            side_effect=lambda *a, **kw: saved.append(a),
        ):
            sampler.run(n_samples=n_samples, nfe_values=list(nfe_values),
                        save_grid=save_grid)
        return saved


def _model(*, sample_clamp: bool):
    model = torch.nn.Linear(1, 1)
    model.cfg = SimpleNamespace(sample_clamp=sample_clamp)
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestFourChannelGuard:

    def test_four_channel_does_not_call_save_image(self):
        """Three-channel normalized latent output must never reach save_image."""
        alg = _LatentAlg(model=_model(sample_clamp=False))
        calls = _run_sampler_run(alg, n_samples=4, nfe_values=[1, 2])
        assert len(calls) == 0, \
            f"save_image was called {len(calls)} times for normalized latents"

    def test_three_channel_calls_save_image_once_per_nfe(self):
        """3-channel RGB output preserves the original save_image behaviour."""
        alg = _PixelAlg(model=_model(sample_clamp=True))
        calls = _run_sampler_run(alg, n_samples=4, nfe_values=[1, 5])
        assert len(calls) == 2, \
            f"save_image called {len(calls)} times (expected once per NFE)"

    def test_two_channel_does_not_call_save_image(self):
        """Any non-RGB channel count is also guarded."""
        alg = _TwoChannelAlg(model=_model(sample_clamp=True))
        calls = _run_sampler_run(alg, n_samples=2, nfe_values=[1])
        assert len(calls) == 0, \
            f"save_image was called for 2-channel output"

    def test_save_grid_false_never_calls_save_image(self):
        """save_grid=False must suppress saving for both pixel and latent output."""
        pixel_alg = _PixelAlg(model=_model(sample_clamp=True))
        latent_alg = _LatentAlg(model=_model(sample_clamp=False))
        for alg in [pixel_alg, latent_alg]:
            calls = _run_sampler_run(alg, n_samples=2, nfe_values=[1], save_grid=False)
            assert len(calls) == 0, \
                f"save_image called with save_grid=False for {type(alg).__name__}"

    def test_four_channel_logs_warning(self, caplog):
        """A warning must be logged when the grid save is suppressed."""
        alg = _LatentAlg(model=_model(sample_clamp=False))
        with caplog.at_level(logging.WARNING, logger="sampling.sampler"):
            _run_sampler_run(alg, n_samples=2, nfe_values=[1])
        # At least one warning must mention the channel count
        warning_texts = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("channel" in w.lower() or "latent" in w.lower()
                   for w in warning_texts), \
            f"No channel-related warning logged. Got: {warning_texts}"

    def test_three_channel_no_warning_for_channel_suppression(self, caplog):
        """RGB pixel output must not trigger the latent suppression warning."""
        alg = _PixelAlg(model=_model(sample_clamp=True))
        with caplog.at_level(logging.WARNING, logger="sampling.sampler"):
            _run_sampler_run(alg, n_samples=2, nfe_values=[1])
        # No warning about channel suppression
        suppression_warnings = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and
               ("channel" in r.message.lower() and "skip" in r.message.lower())
        ]
        assert len(suppression_warnings) == 0, \
            f"Unexpected suppression warning for RGB: {suppression_warnings}"


class TestSamplerGenerateForEvaluation:
    """generate_for_evaluation must pass through latents unchanged (no decoding)."""

    def test_returns_raw_three_channel_latents_for_latent_alg(self):
        alg = _LatentAlg(model=torch.nn.Linear(1, 1))
        with tempfile.TemporaryDirectory() as tmp:
            sampler = Sampler(alg, torch.device("cpu"), tmp, "test", seed=0)
            out = sampler.generate_for_evaluation(n_samples=6, nfe=1, batch_size=4)
        assert out.shape[1:] == (3, 16, 16), \
            f"generate_for_evaluation should return raw VQ latents: {out.shape}"

    def test_returns_three_channel_for_pixel_alg(self):
        alg = _PixelAlg(model=torch.nn.Linear(1, 1))
        with tempfile.TemporaryDirectory() as tmp:
            sampler = Sampler(alg, torch.device("cpu"), tmp, "test", seed=0)
            out = sampler.generate_for_evaluation(n_samples=6, nfe=1, batch_size=4)
        assert out.shape[1] == 3, \
            f"generate_for_evaluation should return 3-ch RGB: {out.shape}"

    def test_total_samples_exact(self):
        alg = _LatentAlg(model=torch.nn.Linear(1, 1))
        with tempfile.TemporaryDirectory() as tmp:
            sampler = Sampler(alg, torch.device("cpu"), tmp, "test", seed=0)
            out = sampler.generate_for_evaluation(n_samples=10, nfe=1, batch_size=3)
        assert out.shape[0] == 10, \
            f"Expected exactly 10 samples, got {out.shape[0]}"
