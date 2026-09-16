#!/usr/bin/env python3
"""Cross-platform, non-training verification for the documented workflow."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

REQUIRED_ALGORITHMS = {
    "fm", "fm_lognorm", "mf", "mf_distill", "consistency", "reflow"
}
CONFIGS = (
    "config/fm_full.json",
    "config/fm_lognorm_full.json",
    "config/mf_full.json",
    "config/mf_distill_full.json",
    "config/consistency_full.json",
    "config/reflow_full.json",
    "config/fm_celeba64.json",
    "config/fm_lognorm_celeba64.json",
    "config/mf_celeba64.json",
    "config/mf_distill_celeba64.json",
    "config/consistency_celeba64.json",
    "config/reflow_celeba64.json",
    "config/mf_coarse16.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate imports, configs, datasets, and workflow prerequisites."
    )
    parser.add_argument(
        "--dataset",
        choices=("none", "cifar10", "celeba"),
        default="none",
        help="Also load and validate one real dataset batch (may download data).",
    )
    parser.add_argument(
        "--strict-prerequisites",
        action="store_true",
        help="Fail if teacher checkpoints or generated Reflow pairs are missing.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    failures: list[str] = []
    blockers: list[str] = []

    from algorithms import ALGORITHM_REGISTRY
    from config.config import ExperimentConfig
    from data.dataset_registry import DATASET_REGISTRY, get_dataloaders_for_config

    missing_algorithms = REQUIRED_ALGORITHMS - set(ALGORITHM_REGISTRY)
    if missing_algorithms:
        failures.append(f"missing algorithms: {sorted(missing_algorithms)}")
    else:
        extras = sorted(set(ALGORITHM_REGISTRY) - REQUIRED_ALGORITHMS)
        print(f"[OK] algorithms: {sorted(REQUIRED_ALGORITHMS)}")
        if extras:
            print(f"[OK] additional utility algorithms: {extras}")

    if not {"cifar10", "celeba"} <= set(DATASET_REGISTRY):
        failures.append(f"dataset registry is incomplete: {sorted(DATASET_REGISTRY)}")
    else:
        print("[OK] dataset registry: cifar10, celeba")

    loaded_configs: dict[str, ExperimentConfig] = {}
    for relative_path in CONFIGS:
        path = PROJECT_ROOT / relative_path
        if not path.is_file():
            failures.append(f"missing config: {relative_path}")
            continue
        try:
            loaded_configs[relative_path] = ExperimentConfig.load(str(path))
            print(f"[OK] config: {relative_path}")
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            failures.append(f"invalid config {relative_path}: {error}")

    prerequisite_fields = (
        ("config/mf_distill_full.json", "teacher_checkpoint", "MF-Distill teacher"),
        ("config/consistency_full.json", "teacher_checkpoint", "Consistency teacher"),
        ("config/reflow_full.json", "pairs_path", "Reflow pairs"),
        ("config/mf_distill_celeba64.json", "teacher_checkpoint", "CelebA MF-Distill teacher"),
        ("config/consistency_celeba64.json", "teacher_checkpoint", "CelebA Consistency teacher"),
        ("config/reflow_celeba64.json", "pairs_path", "CelebA Reflow pairs"),
    )
    for config_path, field, label in prerequisite_fields:
        cfg = loaded_configs.get(config_path)
        if cfg is None:
            continue
        relative_target = cfg.algorithm_kwargs.get(field)
        if not relative_target:
            failures.append(f"{config_path} has no algorithm_kwargs.{field}")
            continue
        target = PROJECT_ROOT / relative_target
        if target.is_file():
            print(f"[OK] {label}: {relative_target}")
        else:
            blockers.append(f"{label} missing: {relative_target}")
            print(f"[BLOCKED] {label}: {relative_target}")

    if args.dataset != "none":
        config_path = (
            "config/smoke_fast.json"
            if args.dataset == "cifar10"
            else "config/fm_celeba64.json"
        )
        cfg = ExperimentConfig.load(str(PROJECT_ROOT / config_path))
        train_loader, _ = get_dataloaders_for_config(cfg)
        images, _ = next(iter(train_loader))
        expected = (3, cfg.dataset.image_size, cfg.dataset.image_size)
        if tuple(images.shape[1:]) != expected:
            failures.append(
                f"{args.dataset} image shape {tuple(images.shape[1:])}, expected {expected}"
            )
        if float(images.min()) < -1.1 or float(images.max()) > 1.1:
            failures.append(
                f"{args.dataset} range [{float(images.min())}, {float(images.max())}]"
            )
        if not failures:
            print(
                f"[OK] {args.dataset} batch: shape={tuple(images.shape)}, "
                f"range=[{float(images.min()):.3f}, {float(images.max()):.3f}]"
            )

    if failures:
        print("\nVerification failed:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    if blockers and args.strict_prerequisites:
        print("\nCode/config checks passed, but the full workflow is blocked:")
        for blocker in blockers:
            print(f"  - {blocker}")
        return 2

    print("\nWorkflow code and configuration checks passed.")
    if blockers:
        print("Long-running stages remain blocked until these artifacts exist:")
        for blocker in blockers:
            print(f"  - {blocker}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
