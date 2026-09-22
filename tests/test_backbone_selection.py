"""CPU-only checks for the selectable frozen CIFAR-10 backbone."""

from pathlib import Path

import pytest
import torch

from config.config import BackboneConfig, ExperimentConfig
from models.backbone import build_backbone, count_parameters
from models.legacy_cifar_backbone import (
    LEGACY_BACKBONE_GIT_BLOB,
    LegacyCifarUNet,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_CONFIGS = (
    "fm.json",
    "fm_lognorm.json",
    "mf.json",
    "mf_distill.json",
    "consistency.json",
    "reflow.json",
)


def _tiny_config(name: str) -> BackboneConfig:
    return BackboneConfig(
        name=name,
        in_channels=3,
        base_channels=16,
        channel_mults=[1, 2],
        num_res_blocks=1,
        time_embed_dim=32,
    )


def test_legacy_backbone_matches_current_state_schema_and_forward() -> None:
    torch.manual_seed(7)
    current = build_backbone(_tiny_config("simple_unet"), image_size=32)
    torch.manual_seed(7)
    legacy = build_backbone(_tiny_config("legacy_cifar_unet"), image_size=32)

    assert isinstance(legacy, LegacyCifarUNet)
    assert LEGACY_BACKBONE_GIT_BLOB == "d1b02c84feba4f1f4360a7adf54c3a98729824c8"
    assert list(current.state_dict()) == list(legacy.state_dict())
    assert count_parameters(current) == count_parameters(legacy)

    legacy.load_state_dict(current.state_dict(), strict=True)
    x = torch.randn(2, 3, 32, 32)
    t = torch.tensor([0.25, 0.75])
    current.eval()
    legacy.eval()
    with torch.no_grad():
        assert torch.equal(current(x, t), legacy(x, t))


def test_legacy_backbone_rejects_non_cifar_shape() -> None:
    with pytest.raises(ValueError, match="3x32x32"):
        build_backbone(_tiny_config("legacy_cifar_unet"), image_size=64)


def test_all_legacy_cifar_presets_are_isolated_and_parse() -> None:
    for filename in LEGACY_CONFIGS:
        cfg = ExperimentConfig.load(
            str(PROJECT_ROOT / "config" / "cifar_legacy" / filename)
        )
        assert cfg.dataset.name == "cifar10"
        assert cfg.dataset.image_size == 32
        assert cfg.backbone.name == "legacy_cifar_unet"
        assert cfg.experiment_name.endswith("_legacy_backbone")

    distill = ExperimentConfig.load(
        str(PROJECT_ROOT / "config/cifar_legacy/mf_distill.json")
    )
    consistency = ExperimentConfig.load(
        str(PROJECT_ROOT / "config/cifar_legacy/consistency.json")
    )
    reflow = ExperimentConfig.load(
        str(PROJECT_ROOT / "config/cifar_legacy/reflow.json")
    )
    assert "fm_legacy_backbone_cifar10" in distill.algorithm_kwargs["teacher_checkpoint"]
    assert "fm_legacy_backbone_cifar10" in consistency.algorithm_kwargs["teacher_checkpoint"]
    assert reflow.algorithm_kwargs["pairs_path"].endswith(
        "reflow_pairs_cifar10_legacy_backbone.pt"
    )
