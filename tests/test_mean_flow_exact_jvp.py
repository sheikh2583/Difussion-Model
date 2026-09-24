"""CPU-only diagnostics for the optional Mean Flow exact-JVP path."""

from __future__ import annotations

import json
from pathlib import Path
from types import MethodType
from unittest.mock import patch

import pytest
import torch

from algorithms.mean_flow import MeanFlowAlgorithm
from algorithms.r_embed import LegacyREmbed
from config.config import BackboneConfig
from models.backbone import SimpleUNet
from utils.checkpoints import load_algorithm_state


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TinyModel(SimpleUNet):
    def __init__(self, channels: int = 2, image_size: int = 4):
        super().__init__(
            BackboneConfig(
                in_channels=channels,
                base_channels=8,
                channel_mults=[1],
                num_res_blocks=1,
                time_embed_dim=8,
            )
        )
        self._expected_image_size = image_size


def make_algorithm(**kwargs) -> MeanFlowAlgorithm:
    return MeanFlowAlgorithm(TinyModel(), algorithm_kwargs=kwargs)


@pytest.mark.parametrize("bare_keys", [False, True])
def test_legacy_rembed_checkpoint_restores_historical_forward(bare_keys):
    algorithm = make_algorithm()
    legacy = LegacyREmbed(algorithm.model.cfg.in_channels)
    state = legacy.state_dict()
    if bare_keys:
        state = {key.removeprefix("net."): value for key, value in state.items()}
    checkpoint = {
        "model_state": algorithm.model.state_dict(),
        "extra_module_0_state": state,
    }
    z = torch.randn(3, 2, 4, 4)
    r = torch.rand(3)
    t = torch.rand(3)
    expected = algorithm.model(legacy(z, r), t)

    load_algorithm_state(algorithm, checkpoint)

    assert isinstance(algorithm.trainable_modules()[1], LegacyREmbed)
    assert torch.allclose(algorithm._forward(z, r, t), expected)


def test_exact_jvp_matches_analytic_total_derivative():
    algo = make_algorithm(use_exact_jvp=True)

    def analytic_forward(self, z, r, t):
        return z * t[:, None, None, None] + r[:, None, None, None]

    algo._forward = MethodType(analytic_forward, algo)
    z = torch.randn(3, 2, 4, 4, dtype=torch.float64)
    v = torch.randn_like(z)
    r = torch.rand(3, dtype=torch.float64)
    t = torch.rand(3, dtype=torch.float64)

    dudt = algo._jvp_exact(z, r, t, v)

    z_f, v_f, r_f, t_f = z.float(), v.float(), r.float(), t.float()
    expected_dudt = v_f * t_f[:, None, None, None] + z_f
    assert dudt.dtype == torch.float32
    assert dudt.requires_grad is False
    assert torch.allclose(dudt, expected_dudt, atol=1e-6)


def test_exact_derivative_route_disables_autocast_and_backpropagates():
    algo = make_algorithm(use_exact_jvp=True, p_fd_step=1.0, p_same=0.0)
    batch = torch.randn(3, 2, 4, 4)

    with patch.object(algo, "_jvp_exact", wraps=algo._jvp_exact) as exact_spy, \
         patch("algorithms.mean_flow.torch.autocast", wraps=torch.autocast) as autocast_spy:
        loss = algo.training_step(batch)["loss"]
        loss.backward()

    exact_spy.assert_called_once()
    autocast_spy.assert_called_once_with(device_type="cpu", enabled=False)
    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in algo.model.parameters())
    assert all(parameter.grad is not None for parameter in algo.r_cond.parameters())


def test_exact_jvp_uses_separate_trainable_forward_and_detached_derivative():
    algo = make_algorithm(use_exact_jvp=True, p_fd_step=1.0, p_same=0.0)
    grad_modes = []
    original_forward = algo._forward

    def recording_forward(self, z, r, t):
        grad_modes.append(torch.is_grad_enabled())
        return original_forward(z, r, t)

    algo._forward = MethodType(recording_forward, algo)
    loss = algo.training_step(torch.randn(2, 2, 4, 4))["loss"]
    loss.backward()

    assert grad_modes[0] is True
    assert False in grad_modes[1:]
    assert any(parameter.grad is not None for parameter in algo.model.parameters())
    assert all(parameter.grad is not None for parameter in algo.r_cond.parameters())


def test_default_derivative_route_remains_finite_difference():
    algo = make_algorithm(p_fd_step=1.0, p_same=0.0)
    assert algo.use_exact_jvp is False

    with patch.object(
        algo,
        "_jvp_exact",
        side_effect=AssertionError("exact JVP must remain opt-in"),
    ):
        loss = algo.training_step(torch.randn(2, 2, 4, 4))["loss"]

    assert torch.isfinite(loss)


def test_diagonal_route_is_unchanged_when_exact_jvp_enabled():
    algo = make_algorithm(use_exact_jvp=True, p_fd_step=0.0, p_same=0.0)

    with patch.object(
        algo,
        "_jvp_exact",
        side_effect=AssertionError("diagonal route must not call JVP"),
    ):
        loss = algo.training_step(torch.randn(2, 2, 4, 4))["loss"]

    assert torch.isfinite(loss)


def test_use_exact_jvp_requires_json_boolean():
    with pytest.raises(TypeError, match="must be a boolean"):
        make_algorithm(use_exact_jvp="true")


def test_fd_force_fp32_defaults_false_and_requires_json_boolean():
    assert make_algorithm().fd_force_fp32 is False
    with pytest.raises(TypeError, match="must be a boolean"):
        make_algorithm(fd_force_fp32=1)


def test_fd_force_fp32_disables_outer_autocast_for_all_forwards():
    algo = make_algorithm(fd_force_fp32=True, p_fd_step=1.0, p_same=0.0)
    observations = []
    original_forward = algo._forward

    def recording_forward(self, z, r, t):
        observations.append(
            (z.dtype, r.dtype, t.dtype, torch.is_autocast_enabled("cpu"))
        )
        return original_forward(z, r, t)

    algo._forward = MethodType(recording_forward, algo)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        loss = algo.training_step(torch.randn(2, 2, 4, 4))["loss"]

    assert loss.dtype == torch.float32
    assert len(observations) == 2
    assert all(item == (torch.float32, torch.float32, torch.float32, False)
               for item in observations)


def test_finite_difference_matches_analytic_directional_derivative():
    algo = make_algorithm(jvp_delta_start=1e-3, jvp_delta_end=1e-3)

    def analytic_forward(self, z, r, t):
        return z * t[:, None, None, None] + r[:, None, None, None]

    algo._forward = MethodType(analytic_forward, algo)
    z = torch.randn(3, 2, 4, 4)
    v = torch.randn_like(z)
    r = torch.tensor([0.1, 0.2, 0.3])
    t = torch.tensor([0.4, 0.5, 0.6])
    _, dudt, _ = algo._finite_difference(z, r, t, v)
    expected = v * (t + algo.jvp_delta)[:, None, None, None] + z

    # This forward difference is exact for the z*t bilinear test function up
    # to the h*v cross term included above.
    assert torch.allclose(dudt, expected, atol=2e-4, rtol=2e-4)


def test_finite_difference_uses_one_h_for_z_time_and_division():
    algo = make_algorithm(jvp_delta_start=2e-3, jvp_delta_end=2e-3)
    calls = []

    def recording_forward(self, z, r, t):
        calls.append((z.detach().clone(), t.detach().clone()))
        return z + t[:, None, None, None]

    algo._forward = MethodType(recording_forward, algo)
    z = torch.randn(2, 2, 4, 4)
    v = torch.randn_like(z)
    r = torch.tensor([0.1, 0.4])
    t = torch.tensor([0.3, 1.0])
    _, dudt, h = algo._finite_difference(z, r, t, v)

    assert torch.allclose(calls[1][1] - calls[0][1], h)
    assert torch.allclose(
        calls[1][0], calls[0][0] + h[:, None, None, None] * v
    )
    assert torch.allclose(dudt, v + 1.0, atol=2e-4, rtol=2e-4)


def test_finite_difference_is_finite_at_upper_time_boundary():
    algo = make_algorithm(jvp_delta_start=1e-3, jvp_delta_end=1e-3)
    z = torch.randn(2, 2, 4, 4)
    v = torch.randn_like(z)
    r = torch.tensor([0.5, 1.0])
    t = torch.ones(2)

    _, dudt, h = algo._finite_difference(z, r, t, v)

    assert torch.isfinite(dudt).all()
    assert torch.isfinite(h).all()
    assert (h != 0).all()
    assert t[0] + h[0] >= r[0]


def test_exact_config_only_changes_variant_and_exact_jvp_flag():
    base = json.loads((PROJECT_ROOT / "config/mf_full_v2.json").read_text(encoding="utf-8"))
    exact = json.loads(
        (PROJECT_ROOT / "config/mf_v2_exact_jvp.json").read_text(encoding="utf-8")
    )

    assert exact["experiment_name"] == "mf_v2_exactjvp"
    assert exact["algorithm_kwargs"].pop("use_exact_jvp") is True
    exact["experiment_name"] = base["experiment_name"]
    assert exact == base


def test_v3_batch128_configs_parse_and_are_matched():
    exact_path = PROJECT_ROOT / "config/mf_v3_exact_jvp_b128.json"
    fd_path = PROJECT_ROOT / "config/mf_v3_fd_fp32_b128.json"
    from config.config import ExperimentConfig

    exact_cfg = ExperimentConfig.load(str(exact_path))
    fd_cfg = ExperimentConfig.load(str(fd_path))
    for cfg in (exact_cfg, fd_cfg):
        assert (cfg.batch_size, cfg.epochs, cfg.seed) == (128, 100, 0)
        assert cfg.optim.learning_rate == 1e-4
        assert cfg.optim.weight_decay == 1e-4
        assert cfg.optim.gradient_clip_norm == 1.0
        assert cfg.optim.scheduler == "cosine"
        assert cfg.optim.scheduler_kwargs == {"eta_min": 1e-6}
        assert cfg.evaluation.num_generated_samples == 5000
        assert cfg.evaluation.nfe_values == [1, 2, 5, 10, 20]

    exact = json.loads(exact_path.read_text(encoding="utf-8"))
    fd = json.loads(fd_path.read_text(encoding="utf-8"))
    assert exact["algorithm_kwargs"]["use_exact_jvp"] is True
    assert exact["algorithm_kwargs"]["fd_force_fp32"] is False
    assert fd["algorithm_kwargs"]["use_exact_jvp"] is False
    assert fd["algorithm_kwargs"]["fd_force_fp32"] is True
    exact["experiment_name"] = fd["experiment_name"]
    for key in ("use_exact_jvp", "fd_force_fp32"):
        exact["algorithm_kwargs"].pop(key)
        fd["algorithm_kwargs"].pop(key)
    assert exact == fd
