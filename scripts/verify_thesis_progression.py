#!/usr/bin/env python3
"""Audit whether every thesis stage is reconstructable from local evidence.

This is deliberately static/read-only: it parses configs and evidence files,
but never starts training, evaluation, downloads, or GPU work.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from config.config import ExperimentConfig


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Stage:
    number: int
    name: str
    status: str
    required: tuple[str, ...]
    note: str


STAGES = (
    Stage(1, "Shared SimpleUNet scaffold", "complete", (
        "models/backbone.py", "training/trainer.py", "experiments/runner.py",
    ), "One shared time-conditioned U-Net and generic train/eval/log pipeline."),
    Stage(2, "CIFAR-10 first three objectives", "complete", (
        "config/fm_full.json", "config/fm_lognorm_full.json", "config/mf_full.json",
        "results/fm_cifar10/metrics/fm_cifar10.jsonl",
        "results/fm_lognorm_cifar10/metrics/fm_lognorm_cifar10.jsonl",
        "results/mf_cifar10/metrics/mf_cifar10.jsonl",
        "results/fm_cifar10/checkpoints/run_1/FlowMatchingAlgorithm_epoch100.pt",
        "results/fm_lognorm_cifar10/checkpoints/run_1/FlowMatchingLognormAlgorithm_epoch100.pt",
        "results/mf_cifar10/checkpoints/run_1/MeanFlowAlgorithm_epoch100.pt",
    ), "FM, FM-LN, and Mean Flow establish the first algorithm progression."),
    Stage(3, "CIFAR-10 six-method comparison", "complete", (
        "config/mf_distill_full.json", "config/consistency_full.json",
        "config/reflow_full.json",
        "results/mf_distill_cifar10/metrics/mf_distill_cifar10.jsonl",
        "results/consistency_cifar10/metrics/consistency_cifar10.jsonl",
        "results/reflow_cifar10/metrics/reflow_cifar10.jsonl",
        "results/mf_distill_cifar10/checkpoints/run_2/MeanFlowDistillAlgorithm_epoch100.pt",
        "results/consistency_cifar10/checkpoints/run_1/ConsistencyAlgorithm_epoch100.pt",
        "results/reflow_cifar10/checkpoints/run_1/ReflowAlgorithm_epoch100.pt",
    ), "Adds MF-Distill, Consistency, and Reflow under the shared backbone."),
    Stage(4, "CelebA 64x64 pixel transfer", "partial", (
        "config/fm_celeba64.json", "config/fm_lognorm_celeba64.json",
        "config/mf_celeba64.json", "config/mf_distill_celeba64.json",
        "config/consistency_celeba64.json", "config/reflow_celeba64.json",
        "results/fm_celeba/metrics/fm_celeba.jsonl",
        "results/fm_lognorm_celeba/metrics/fm_lognorm_celeba.jsonl",
        "results/fm_celeba/checkpoints/run_2/FlowMatchingAlgorithm_epoch100.pt",
        "results/fm_lognorm_celeba/checkpoints/run_2/FlowMatchingLognormAlgorithm_epoch100.pt",
    ), "All six are runnable; only FM and FM-LN have completed measured evidence."),
    Stage(5, "CelebA frozen-latent migration", "complete", (
        "results/codecs/celeba_vq_f4/accepted_codec.pt",
        "config/fm_celeba_latent.json", "config/fm_lognorm_celeba_latent.json",
        "data/reflow_pairs_celeba_latent.pt",
        "results/fm_celeba_latent/checkpoints/run_4/FlowMatchingAlgorithm_epoch100.pt",
        "results/fm_lognorm_celeba_latent/checkpoints/run_2/FlowMatchingLognormAlgorithm_epoch100.pt",
        "results/mf_celeba_latent/checkpoints/run_2/MeanFlowAlgorithm_epoch80.pt",
        "results/mf_celeba_latent/checkpoints/run_2/MeanFlowAlgorithm_epoch100.pt",
        "results/mf_distill_celeba_latent/checkpoints/run_1/MeanFlowDistillAlgorithm_epoch100.pt",
        "results/consistency_celeba_latent/checkpoints/run_1/ConsistencyAlgorithm_epoch90.pt",
        "results/reflow_celeba_latent/checkpoints/run_1/ReflowAlgorithm_epoch100.pt",
    ), "Moves 3x64x64 RGB generation to normalized 3x16x16 VQ-f4 states."),
    Stage(6, "Latent fixes and Hutchinson diagnostic", "partial", (
        "algorithms/mean_flow_hutchinson.py",
        "config/mf_hutchinson_cv_celeba_latent.json",
        "results/mf_hutchinson_cv_celeba_latent/metrics/mf_hutchinson_cv_celeba_latent.jsonl",
        "results/mf_hutchinson_cv_celeba_latent/checkpoints/run_1/MeanFlowHutchinsonAlgorithm_epoch90.pt",
    ), "The CV run has evidence through epoch 91 and an epoch-90 recovery checkpoint."),
    Stage(7, "Latent inference extensions", "implementation-only", (
        "algorithms/mean_flow_adaptive_nfe.py",
        "algorithms/mean_flow_multiscale.py",
        "scripts/sample_mean_flow_extensions.py",
        "tests/test_latent_extensions.py",
    ), "Adaptive NFE and single-model coarse/refine are templates, not trained methods."),
    Stage(8, "Higher-resolution face generation", "complete", (
        "config/fm_celeba64.json", "results/fm_celeba/checkpoints",
        "results/codecs/celeba_vq_f4/validation_report.json",
    ), "High resolution here means decoded 64x64 CelebA versus 32x32 CIFAR-10."),
    Stage(9, "Evaluation and comparison narrative", "complete", (
        "results/aggregate/algorithm_progression_vs_fm.csv",
        "results/aggregate/cross_dataset_consistency.csv",
        "results/aggregate/pixel_vs_latent.csv",
        "results/aggregate/quality_compute_pareto.csv",
        "results/aggregate/comparison_manifest.json",
        "docs/COMPARISON_PROTOCOL.md",
    ), "FID/IS, compute, transfer, representation, and provenance remain separated."),
)


def _nonempty_jsonl(relative: str) -> bool:
    path = ROOT / relative
    if path.suffix != ".jsonl":
        return path.exists()
    if not path.is_file():
        return False
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            if isinstance(json.loads(line), dict):
                return True
        except json.JSONDecodeError:
            continue
    return False


def main() -> int:
    missing: list[str] = []
    print("Thesis progression reconstruction audit")
    print("=" * 40)
    for stage in STAGES:
        absent = [path for path in stage.required if not _nonempty_jsonl(path)]
        marker = "OK" if not absent else "MISSING"
        print(f"[{marker}] Stage {stage.number}: {stage.name} ({stage.status})")
        print(f"     {stage.note}")
        for path in absent:
            print(f"     missing: {path}")
            missing.append(path)

    # Every experiment preset must continue to parse, including historical
    # and implementation-diagnostic variants.
    config_errors = []
    for path in sorted((ROOT / "config").rglob("*.json")):
        try:
            ExperimentConfig.load(str(path))
        except Exception as error:  # pragma: no cover - defensive CLI report
            config_errors.append(f"{path.relative_to(ROOT)}: {error}")
    if config_errors:
        print("[MISSING] Config parsing")
        for error in config_errors:
            print(f"     {error}")
    else:
        print("[OK] All experiment configs parse")

    if missing or config_errors:
        print("Audit failed: required reconstruction evidence is missing.")
        return 1
    print("Audit passed: all declared stages have their required local evidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
