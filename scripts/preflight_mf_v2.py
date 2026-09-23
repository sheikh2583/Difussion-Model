#!/usr/bin/env python3
"""Mean Flow safety checks for controlled v2/v3 experiments.

The default v2 preflight is static and never imports torch. ``--verify-v3``
also runs a tiny CPU-only derivative/gradient check; it never loads data or
checkpoints and does not run an optimizer.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PreflightError(ValueError):
    pass


def verify_mf_v3(project_root: Path = PROJECT_ROOT) -> None:
    """Verify the paired v3 configs and stabilized derivative paths on CPU."""
    import torch

    from algorithms.mean_flow import MeanFlowAlgorithm
    from config.config import BackboneConfig, ExperimentConfig
    from models.backbone import SimpleUNet

    exact_path = project_root / "config/mf_v3_exact_jvp_b128.json"
    fd_path = project_root / "config/mf_v3_fd_fp32_b128.json"
    exact_cfg = ExperimentConfig.load(str(exact_path))
    fd_cfg = ExperimentConfig.load(str(fd_path))
    errors: list[str] = []
    for path, cfg in ((exact_path, exact_cfg), (fd_path, fd_cfg)):
        _require(errors, cfg.dataset.name == "cifar10", f"{path}: expected CIFAR-10")
        _require(errors, cfg.batch_size == 128, f"{path}: batch size must be 128")
        _require(errors, cfg.epochs == 100, f"{path}: epochs must be 100")
        _require(
            errors,
            cfg.optim.gradient_clip_norm == 1.0,
            f"{path}: gradient clip norm must be 1.0",
        )

    exact = json.loads(exact_path.read_text(encoding="utf-8"))
    finite_difference = json.loads(fd_path.read_text(encoding="utf-8"))
    _require(errors, exact["algorithm_kwargs"].get("use_exact_jvp") is True,
             f"{exact_path}: use_exact_jvp must be true")
    _require(errors, exact["algorithm_kwargs"].get("fd_force_fp32") is False,
             f"{exact_path}: fd_force_fp32 must be false")
    _require(errors, finite_difference["algorithm_kwargs"].get("use_exact_jvp") is False,
             f"{fd_path}: use_exact_jvp must be false")
    _require(errors, finite_difference["algorithm_kwargs"].get("fd_force_fp32") is True,
             f"{fd_path}: fd_force_fp32 must be true")
    exact["experiment_name"] = finite_difference["experiment_name"]
    for key in ("use_exact_jvp", "fd_force_fp32"):
        exact["algorithm_kwargs"].pop(key)
        finite_difference["algorithm_kwargs"].pop(key)
    _require(
        errors,
        exact == finite_difference,
        "controlled v3 configs differ outside experiment name/JVP strategy",
    )
    if errors:
        raise PreflightError("MF v3 verification failed:\n- " + "\n- ".join(errors))

    def tiny_algorithm(**kwargs: Any) -> MeanFlowAlgorithm:
        model = SimpleUNet(
            BackboneConfig(
                in_channels=2,
                base_channels=8,
                channel_mults=[1],
                num_res_blocks=1,
                time_embed_dim=8,
            )
        )
        model._expected_image_size = 4
        return MeanFlowAlgorithm(model, algorithm_kwargs=kwargs)

    torch.manual_seed(0)
    batch = torch.randn(2, 2, 4, 4)
    exact_algorithm = tiny_algorithm(
        use_exact_jvp=True, p_fd_step=1.0, p_same=0.0
    )
    loss = exact_algorithm.training_step(batch)["loss"]
    if not torch.isfinite(loss):
        raise PreflightError("MF v3 exact-JVP synthetic loss is non-finite")
    loss.backward()
    if not any(
        parameter.grad is not None for parameter in exact_algorithm.model.parameters()
    ):
        raise PreflightError("MF v3 exact-JVP route did not reach the backbone")
    if not all(
        parameter.grad is not None for parameter in exact_algorithm.r_cond.parameters()
    ):
        raise PreflightError("MF v3 exact-JVP route did not reach r-conditioning")

    fd_algorithm = tiny_algorithm(
        fd_force_fp32=True,
        p_fd_step=1.0,
        p_same=0.0,
        jvp_delta_start=1e-3,
        jvp_delta_end=1e-3,
    )
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        fd_loss = fd_algorithm.training_step(batch)["loss"]
    if fd_loss.dtype != torch.float32 or not torch.isfinite(fd_loss):
        raise PreflightError("MF v3 forced-FP32 FD loss is not finite float32")
    _, derivative, step = fd_algorithm._finite_difference(
        torch.randn_like(batch),
        torch.tensor([0.5, 1.0]),
        torch.ones(2),
        torch.randn_like(batch),
    )
    if not torch.isfinite(derivative).all() or not torch.isfinite(step).all():
        raise PreflightError("MF v3 finite-difference boundary result is non-finite")
    if (step == 0).any():
        raise PreflightError("MF v3 finite-difference boundary step collapsed to zero")


def _require(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def _same_number(actual: Any, expected: float) -> bool:
    return isinstance(actual, (int, float)) and math.isclose(
        float(actual), expected, rel_tol=0.0, abs_tol=1e-12
    )


def validate_mf_v2(
    raw: dict[str, Any], project_root: Path, strict_evidence: bool = False
) -> list[str]:
    """Validate the controlled experiment and return non-fatal warnings."""
    errors: list[str] = []
    warnings: list[str] = []

    for alias in ("num_epochs", "use_amp", "lr_scheduler"):
        _require(errors, alias not in raw, f"legacy field {alias!r} is not supported")

    optim = raw.get("optim", {})
    evaluation = raw.get("evaluation", {})
    dataset = raw.get("dataset", {})
    kwargs = raw.get("algorithm_kwargs", {})
    _require(errors, "lr" not in optim, "use optim.learning_rate, not optim.lr")
    _require(errors, raw.get("experiment_name") == "mf_v2", "experiment_name must be 'mf_v2'")
    _require(errors, raw.get("epochs") == 100, "full-run scheduler horizon/epochs must be 100")
    _require(errors, raw.get("batch_size") == 128, "batch_size must be 128")
    _require(errors, raw.get("amp") is True, "amp must be true")
    _require(errors, dataset.get("name") == "cifar10", "dataset.name must be 'cifar10'")
    _require(errors, optim.get("optimizer") == "adamw", "optim.optimizer must be 'adamw'")
    _require(errors, _same_number(optim.get("learning_rate"), 1e-4), "learning rate must be 1e-4")
    _require(errors, _same_number(optim.get("weight_decay"), 1e-4), "weight decay must be 1e-4")
    _require(errors, _same_number(optim.get("gradient_clip_norm"), 1.0), "gradient clip norm must be 1.0")
    _require(errors, optim.get("scheduler") == "cosine", "scheduler must be 'cosine'")
    _require(
        errors,
        _same_number(optim.get("scheduler_kwargs", {}).get("eta_min"), 1e-6),
        "cosine eta_min must be 1e-6",
    )
    _require(errors, evaluation.get("num_generated_samples") == 5000, "evaluation sample count must be 5000")
    _require(errors, evaluation.get("nfe_values") == [1, 2, 5, 10, 20], "evaluation NFE grid must be [1,2,5,10,20]")
    _require(errors, _same_number(kwargs.get("p_same"), 0.25), "p_same must be 0.25")
    _require(errors, _same_number(kwargs.get("p_fd_step"), 0.5), "p_fd_step must be 0.5")
    _require(errors, _same_number(kwargs.get("jvp_delta_start"), 1e-3), "jvp_delta_start must be 1e-3")
    _require(errors, _same_number(kwargs.get("jvp_delta_end"), 1e-4), "jvp_delta_end must be 1e-4")

    output = Path(raw.get("output_dir", "./results"))
    if not output.is_absolute():
        output = project_root / output
    old_run = (output / "mf_cifar10").resolve()
    full_run = (output / "mf_v2_cifar10").resolve()
    probe_run = (output / "mf_v2_probe_cifar10").resolve()
    _require(errors, len({old_run, full_run, probe_run}) == 3, "old, full-v2, and probe run directories must be distinct")

    cache = evaluation.get("fid_reference_cache")
    _require(errors, bool(cache), "evaluation.fid_reference_cache is required")
    old_config = project_root / "config" / "mf_full.json"
    if old_config.is_file() and cache:
        old_raw = json.loads(old_config.read_text(encoding="utf-8"))
        old_cache = old_raw.get("evaluation", {}).get("fid_reference_cache")
        _require(errors, cache != old_cache, "MF v2 must use a distinct FID reference cache path")

    if not old_run.is_dir():
        message = (
            f"original MF evidence is absent at {old_run}; this is expected in a "
            "fresh clone because results are not versioned"
        )
        if strict_evidence:
            errors.append(message)
        else:
            warnings.append(message)

    if errors:
        raise PreflightError("MF v2 preflight failed:\n- " + "\n- ".join(errors))
    return warnings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/mf_full_v2.json")
    parser.add_argument(
        "--strict-evidence", action="store_true",
        help="Fail if the local original results/mf_cifar10 evidence is absent.",
    )
    parser.add_argument(
        "--verify-v3", action="store_true",
        help="Run paired-config checks and tiny CPU derivative diagnostics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.verify_v3:
        try:
            verify_mf_v3(PROJECT_ROOT)
        except (PreflightError, ValueError, json.JSONDecodeError) as exc:
            raise SystemExit(str(exc)) from exc
        print(
            "MF v3 verification passed: controlled configs, exact JVP, FP32 "
            "FD boundary handling, and gradient flow are valid on CPU."
        )
        return
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if not config_path.is_file():
        raise SystemExit(f"MF v2 config not found: {config_path}")
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    try:
        warnings = validate_mf_v2(raw, PROJECT_ROOT, args.strict_evidence)
    except (PreflightError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc
    print("MF v2 preflight passed (static checks only; no model was run).")
    print("Full horizon: 100 epochs; probe override: 15 epochs in a separate run.")
    for warning in warnings:
        print(f"WARNING: {warning}")


if __name__ == "__main__":
    main()
