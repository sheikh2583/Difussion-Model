"""CPU-only checks for latent Mean Flow inference wrappers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from algorithms.mean_flow_adaptive_nfe import AdaptiveMeanFlowSampler
from algorithms.mean_flow_multiscale import MultiScaleMeanFlowPipeline


class MockMFAlgorithm:
    def __init__(self, image_size: int = 16):
        self.model = SimpleNamespace(
            cfg=SimpleNamespace(in_channels=3),
            _expected_image_size=image_size,
        )

    def trainable_modules(self):
        return []

    def _forward(self, z, r, t):
        del r, t
        return torch.zeros_like(z)

    def sample(self, n_samples, nfe, device):
        del nfe
        return torch.zeros(n_samples, 3, 16, 16, device=device)


class MockCodec:
    def __init__(self):
        self.maximum_input = None

    def decode_normalised(self, latents):
        self.maximum_input = float(latents.abs().max())
        return latents.repeat_interleave(4, -2).repeat_interleave(4, -1) * 2


@pytest.mark.parametrize(
    "factory",
    [
        lambda algorithm: AdaptiveMeanFlowSampler(algorithm),
        lambda algorithm: MultiScaleMeanFlowPipeline(algorithm),
    ],
)
def test_latent_extensions_reject_non_16_spatial_models(factory):
    wrapper = factory(MockMFAlgorithm(image_size=32))
    with pytest.raises(ValueError, match="expected spatial size 16"):
        wrapper.sample(2, torch.device("cpu"))


def test_adaptive_returns_unbounded_latents_and_tracks_realized_nfe():
    torch.manual_seed(7)
    sampler = AdaptiveMeanFlowSampler(
        MockMFAlgorithm(), min_nfe=1, max_nfe=4, confidence_threshold=0.05
    )

    samples = sampler.sample(8, torch.device("cpu"))

    assert samples.shape == (8, 3, 16, 16)
    assert torch.isfinite(samples).all()
    assert (samples.abs() > 1.0).any()
    assert sampler.last_nfe_per_sample.shape == (8,)
    assert 1 <= sampler.last_average_nfe <= 4


@pytest.mark.parametrize("t_renoise", [0.0, 0.3])
def test_single_model_refinement_returns_valid_latents(t_renoise):
    torch.manual_seed(11)
    pipeline = MultiScaleMeanFlowPipeline(
        MockMFAlgorithm(), coarse_nfe=1, fine_nfe=4, t_renoise=t_renoise
    )

    samples = pipeline.sample(5, torch.device("cpu"))

    assert samples.shape == (5, 3, 16, 16)
    assert torch.isfinite(samples).all()


@pytest.mark.parametrize(
    "factory",
    [
        lambda algorithm, codec: AdaptiveMeanFlowSampler(
            algorithm, max_nfe=2, codec=codec
        ),
        lambda algorithm, codec: MultiScaleMeanFlowPipeline(
            algorithm, coarse_nfe=1, fine_nfe=1, codec=codec
        ),
    ],
)
def test_codec_receives_unclamped_latents_and_output_is_clamped(factory):
    torch.manual_seed(13)
    codec = MockCodec()
    wrapper = factory(MockMFAlgorithm(), codec)

    images = wrapper.sample(8, torch.device("cpu"))

    assert codec.maximum_input > 1.0
    assert images.shape == (8, 3, 64, 64)
    assert images.min() >= -1.0
    assert images.max() <= 1.0
