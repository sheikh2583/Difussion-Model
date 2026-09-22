from __future__ import annotations

from pathlib import Path

import torch

from algorithms.flow_matching import FlowMatchingAlgorithm
from config.config import ExperimentConfig
from train import verified_completed_checkpoint
from utils.checkpoint_provenance import build_provenance


def test_completed_run_skip_requires_matching_provenance(tmp_path: Path) -> None:
    cfg = ExperimentConfig(experiment_name="fm", epochs=3)
    run_dir = tmp_path / "results" / "fm_cifar10"
    checkpoint = run_dir / "checkpoints" / "run_1" / "FlowMatchingAlgorithm_epoch3.pt"
    checkpoint.parent.mkdir(parents=True)
    torch.save({
        "epoch": 3,
        "provenance": build_provenance(cfg, FlowMatchingAlgorithm, "fm"),
    }, checkpoint)

    assert verified_completed_checkpoint(
        run_dir,
        algorithm_cls=FlowMatchingAlgorithm,
        algorithm_key="fm",
        cfg=cfg,
        run_number=1,
    ) == checkpoint
