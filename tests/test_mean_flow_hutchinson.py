"""CPU-only scope checks for the latent-only Hutchinson MF variant."""

import json
from pathlib import Path

import pytest
import torch

from algorithms.mean_flow_hutchinson import MeanFlowHutchinsonAlgorithm
from config.config import BackboneConfig, ExperimentConfig
from models.backbone import build_backbone
from training.trainer import build_optimizer, build_scheduler
from utils.run_lifecycle import run_directory


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


def test_control_variate_recovers_linear_directional_derivative():
    algorithm = MeanFlowHutchinsonAlgorithm(
        _model(latent=True),
        {
            "n_probes": 4,
            "fd_eps_start": 1e-2,
            "fd_eps_end": 1e-4,
        },
    )

    def linear_forward(z, r, t):
        del r
        return 2.0 * z + 3.0 * t.view(-1, 1, 1, 1)

    algorithm._forward = linear_forward
    z = torch.randn(2, 3, 4, 4)
    v = torch.randn_like(z)
    r = torch.tensor([0.1, 0.25])
    t = torch.tensor([0.4, 1.0])

    estimate = algorithm._hutchinson_dudt(z, r, t, v)
    exact = 2.0 * v + 3.0

    assert estimate.requires_grad is False
    assert torch.allclose(estimate, exact, atol=2e-4, rtol=2e-4)


def test_control_variate_epsilon_anneals_with_epoch_progress():
    algorithm = MeanFlowHutchinsonAlgorithm(
        _model(latent=True),
        {"fd_eps_start": 1e-2, "fd_eps_end": 1e-4},
    )

    assert algorithm.fd_eps == pytest.approx(1e-2)
    algorithm.on_epoch_end(50, 100)
    assert algorithm.fd_eps == pytest.approx(0.00505)
    algorithm.on_epoch_end(100, 100)
    assert algorithm.fd_eps == pytest.approx(1e-4)


@pytest.mark.parametrize("key", ["fd_eps_start", "fd_eps_end"])
def test_control_variate_rejects_non_positive_epsilon(key):
    with pytest.raises(ValueError, match="must be positive"):
        MeanFlowHutchinsonAlgorithm(_model(latent=True), {key: 0.0})


def test_control_variate_config_uses_repository_schema_and_isolated_run(tmp_path):
    config_path = PROJECT_ROOT / "config/mf_hutchinson_cv_celeba_latent.json"
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    cfg = ExperimentConfig.load(str(config_path))

    assert "gradient_clip_norm" not in raw
    assert cfg.optim.gradient_clip_norm == 1.0
    assert cfg.optim.scheduler == "cosine"
    assert cfg.optim.scheduler_kwargs == {"eta_min": 1e-6}
    assert cfg.algorithm_kwargs == {
        "p_same": 0.1,
        "p_hutchinson_step": 0.8,
        "n_probes": 4,
        "fd_eps_start": 0.01,
        "fd_eps_end": 0.0001,
    }
    assert run_directory(cfg, tmp_path).name == "mf_hutchinson_cv_celeba_latent"

    optimizer = build_optimizer([_model(latent=True)], cfg)
    scheduler = build_scheduler(optimizer, cfg)
    assert scheduler.T_max == 100


def test_control_variate_smoke_no_loss_divergence():
    """Run 20 training steps on synthetic data and verify loss stays bounded.

    The raw Hutchinson estimator (pre-CV) produced losses of 185-238.
    With the FD control variate, losses should stay in the single-digit
    range even on random data with a randomly initialized model.
    """
    torch.manual_seed(42)
    algorithm = MeanFlowHutchinsonAlgorithm(
        _model(latent=True),
        {
            "p_same": 0.1,
            "p_hutchinson_step": 0.8,
            "n_probes": 4,
            "fd_eps_start": 1e-2,
            "fd_eps_end": 1e-4,
        },
    )

    params = []
    for mod in algorithm.trainable_modules():
        params.extend(mod.parameters())
    optimizer = torch.optim.AdamW(params, lr=5e-4)

    losses = []
    for _ in range(20):
        optimizer.zero_grad()
        result = algorithm.training_step(torch.randn(4, 3, 8, 8))
        loss = result["loss"]
        assert torch.isfinite(loss), f"Non-finite loss: {loss.item()}"
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # No explosion: final/initial ratio should be well below 5x.
    # Raw Hutchinson had 185/2 ≈ 90x.
    ratio = losses[-1] / max(losses[0], 1e-8)
    assert ratio < 5.0, f"Loss diverging: final/initial = {ratio:.2f}"

    # No sustained upward trend in last 10 steps
    last_10 = losses[-10:]
    upward = sum(1 for i in range(1, len(last_10)) if last_10[i] > last_10[i - 1])
    assert upward < 8, f"Sustained upward trend: {upward}/9 steps increasing"
