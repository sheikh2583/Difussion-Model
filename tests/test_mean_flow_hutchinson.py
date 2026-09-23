"""CPU-only scope checks for the latent-only Hutchinson MF variant."""

import pytest

from algorithms.mean_flow_hutchinson import MeanFlowHutchinsonAlgorithm
from config.config import BackboneConfig
from models.backbone import build_backbone


def _model(*, latent: bool):
    cfg = BackboneConfig(
        in_channels=3,
        base_channels=8,
        channel_mults=[1],
        num_res_blocks=1,
        time_embed_dim=16,
        sample_clamp=not latent,
    )
    return build_backbone(cfg, image_size=8)


def test_hutchinson_rejects_pixel_space_backbone():
    with pytest.raises(ValueError, match="restricted to latent-space"):
        MeanFlowHutchinsonAlgorithm(_model(latent=False))


def test_hutchinson_accepts_latent_space_backbone():
    algorithm = MeanFlowHutchinsonAlgorithm(_model(latent=True))
    assert algorithm.model.cfg.sample_clamp is False
