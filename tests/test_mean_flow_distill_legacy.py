"""MF-Distill must load early latent checkpoints that used LegacyREmbed."""
import torch

from algorithms.mean_flow_distill import MeanFlowDistillAlgorithm
from algorithms.r_embed import LegacyREmbed
from config.config import BackboneConfig
from models.backbone import build_backbone
from utils.checkpoints import load_algorithm_state


def _algorithm():
    cfg = BackboneConfig(in_channels=3, base_channels=16, channel_mults=[1, 2],
                         num_res_blocks=1, time_embed_dim=32, sample_clamp=False)
    return MeanFlowDistillAlgorithm(
        build_backbone(cfg, image_size=16), {"teacher_checkpoint": "unused.pt"})


def test_legacy_conditioner_checkpoint_loads_and_samples():
    source = _algorithm()
    legacy = LegacyREmbed(3)
    checkpoint = {
        "model_state": source.model.state_dict(),
        "extra_module_0_state": legacy.state_dict(),
    }
    target = _algorithm()
    load_algorithm_state(target, checkpoint)

    assert target._legacy_r_embed is not None
    assert target.trainable_modules()[1] is target._legacy_r_embed
    for key, value in legacy.state_dict().items():
        assert torch.equal(target._legacy_r_embed.state_dict()[key], value)
    out = target.sample(2, 2, torch.device("cpu"))
    assert out.shape == (2, 3, 16, 16) and torch.isfinite(out).all()


def test_current_checkpoint_keeps_rcond():
    algo = _algorithm()
    checkpoint = {"model_state": algo.model.state_dict(),
                  "extra_module_0_state": algo.r_cond.state_dict()}
    target = _algorithm()
    load_algorithm_state(target, checkpoint)
    assert target._legacy_r_embed is None
