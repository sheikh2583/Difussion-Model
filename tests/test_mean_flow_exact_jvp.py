"""CPU-only diagnostics for the optional Mean Flow exact-JVP path."""

from __future__ import annotations

import json
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from algorithms.mean_flow import MeanFlowAlgorithm


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TinyModel(nn.Module):
    def __init__(self, channels: int = 2, image_size: int = 4):
        super().__init__()
        self.cfg = SimpleNamespace(in_channels=channels)
        self._expected_image_size = image_size
        self.scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.scale * x + t[:, None, None, None]


def make_algorithm(**kwargs) -> MeanFlowAlgorithm:
    return MeanFlowAlgorithm(TinyModel(), algorithm_kwargs=kwargs)


def test_exact_jvp_matches_analytic_total_derivative():
    algo = make_algorithm(use_exact_jvp=True)

    def analytic_forward(self, z, r, t):
        return z * t[:, None, None, None] + r[:, None, None, None]

    algo._forward = MethodType(analytic_forward, algo)
    z = torch.randn(3, 2, 4, 4, dtype=torch.float64)
    v = torch.randn_like(z)
    r = torch.rand(3, dtype=torch.float64)
    t = torch.rand(3, dtype=torch.float64)

    u_pred, dudt = algo._jvp_exact(z, r, t, v)

    z_f, v_f, r_f, t_f = z.float(), v.float(), r.float(), t.float()
    expected_u = z_f * t_f[:, None, None, None] + r_f[:, None, None, None]
    expected_dudt = v_f * t_f[:, None, None, None] + z_f
    assert u_pred.dtype == torch.float32
    assert dudt.dtype == torch.float32
    assert torch.allclose(u_pred, expected_u, atol=1e-6)
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
    assert algo.model.scale.grad is not None


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


def test_exact_config_only_changes_variant_and_exact_jvp_flag():
    base = json.loads((PROJECT_ROOT / "config/mf_full_v2.json").read_text(encoding="utf-8"))
    exact = json.loads(
        (PROJECT_ROOT / "config/mf_v2_exact_jvp.json").read_text(encoding="utf-8")
    )

    assert exact["experiment_name"] == "mf_v2_exactjvp"
    assert exact["algorithm_kwargs"].pop("use_exact_jvp") is True
    exact["experiment_name"] = base["experiment_name"]
    assert exact == base
