"""CPU-only checks for the pre-dataset training startup summary."""

from pathlib import Path

import pytest

from config.config import ExperimentConfig
from train import print_startup_summary
from utils.algorithm_compatibility import validate_algorithm_dataset
from utils.run_environment import add_config_hash


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_startup_summary_uses_existing_environment_identity(capsys, tmp_path):
    config_path = PROJECT_ROOT / "config/mf_v3_exact_jvp_b128.json"
    cfg = ExperimentConfig.load(str(config_path))
    environment = add_config_hash(
        {
            "machine_label": "linux-24gb",
            "code_identity": "abc123+dirty:456def",
            "session_id": "shared-session",
        },
        cfg,
    )

    print_startup_summary(
        config_path=str(config_path),
        algorithm_key="mf",
        cfg=cfg,
        result_dir=tmp_path / "results" / "mf_v3_exact_jvp_b128_cifar10",
        run_environment=environment,
    )
    output = capsys.readouterr().out

    assert f"config: {config_path.resolve()}" in output
    assert "algorithm: mf" in output
    assert "experiment: mf_v3_exact_jvp_b128" in output
    assert f"result_dir: {(tmp_path / 'results' / 'mf_v3_exact_jvp_b128_cifar10').resolve()}" in output
    assert "batch_size: 128" in output
    assert "epochs: 100" in output
    assert "seed: 0" in output
    assert "learning_rate: 0.0001" in output
    assert "scheduler: cosine" in output
    assert "gradient_clip_norm: 1.0" in output
    assert "use_exact_jvp: True" in output
    assert "fd_force_fp32: False" in output
    assert "p_same: 0.25" in output
    assert "p_fd_step: 0.5" in output
    assert "jvp_delta_range: 0.001 -> 0.0001" in output
    assert "machine_label: linux-24gb" in output
    assert "code_identity: abc123+dirty:456def" in output


def test_config_hash_enrichment_does_not_mutate_environment():
    cfg = ExperimentConfig()
    original = {"session_id": "same-session"}

    enriched = add_config_hash(original, cfg)

    assert "config_sha256" not in original
    assert enriched["session_id"] == "same-session"
    assert len(enriched["config_sha256"]) == 64


def test_mf_hutchinson_is_restricted_to_latent_space():
    latent_cfg = ExperimentConfig.load(
        str(PROJECT_ROOT / "config/mf_hutchinson_celeba_latent.json")
    )
    validate_algorithm_dataset("mf_hutchinson", latent_cfg)

    pixel_cfg = ExperimentConfig()
    pixel_cfg.dataset.name = "celeba"
    with pytest.raises(ValueError, match="latent-only"):
        validate_algorithm_dataset("mf_hutchinson", pixel_cfg)


def test_mf_hutchinson_has_no_pixel_preset():
    assert not (PROJECT_ROOT / "config/mf_hutchinson_celeba.json").exists()
