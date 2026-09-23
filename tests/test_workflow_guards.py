from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from utils.gpu_lock import (
    ActiveGpuLockError,
    StaleGpuLockError,
    acquire_gpu_lock,
    recover_stale_lock,
)
from utils.source_identity import (
    SourceIdentityError,
    build_source_manifest,
    verify_source_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_active_lock_rejected_and_nested_token_does_not_release(tmp_path: Path) -> None:
    path = tmp_path / ".lock"
    owner = acquire_gpu_lock(path, command="parent")
    try:
        with pytest.raises(ActiveGpuLockError):
            acquire_gpu_lock(path, command="competitor", inherited_token="wrong")
        child = acquire_gpu_lock(path, command="child", inherited_token=owner.token)
        assert child.owned is False
        assert child.release() is False
        assert path.is_file()
    finally:
        assert owner.release() is True
    assert not path.exists()


def test_stale_lock_requires_explicit_recovery(tmp_path: Path) -> None:
    path = tmp_path / ".lock"
    path.write_text(json.dumps({
        "token": "stale", "pid": 999999999, "hostname": __import__("socket").gethostname(),
        "process_start_ticks": 1, "command": "old", "acquired_utc": "old",
    }), encoding="utf-8")
    with pytest.raises(StaleGpuLockError, match="lock-recover"):
        acquire_gpu_lock(path, command="new")
    assert recover_stale_lock(path) is True
    assert not path.exists()


def _identity_tree(root: Path) -> None:
    for directory in ("algorithms", "models", "training", "data", "evaluation", "sampling"):
        target = root / directory / "unit.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("VALUE = 1\n", encoding="utf-8")
    (root / "config").mkdir()
    (root / "config" / "job.json").write_text("{}\n", encoding="utf-8")


def test_source_change_is_named_but_documentation_change_is_tolerated(tmp_path: Path) -> None:
    _identity_tree(tmp_path)
    manifest = build_source_manifest(tmp_path, configs=["config/job.json"])
    docs = tmp_path / "docs" / "notes.md"
    docs.parent.mkdir()
    docs.write_text("documentation only\n", encoding="utf-8")
    verify_source_manifest(tmp_path, manifest)
    changed = tmp_path / "models" / "unit.py"
    changed.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(SourceIdentityError, match="models/unit.py"):
        verify_source_manifest(tmp_path, manifest)


def test_source_addition_is_rejected(tmp_path: Path) -> None:
    _identity_tree(tmp_path)
    manifest = build_source_manifest(tmp_path, configs=["config/job.json"])
    added = tmp_path / "algorithms" / "new_path.py"
    added.write_text("ENABLED = True\n", encoding="utf-8")

    with pytest.raises(SourceIdentityError, match=r"algorithms/new_path\.py \(added\)"):
        verify_source_manifest(tmp_path, manifest)


def test_latent_suite_dry_run_is_serialized_in_dependency_order() -> None:
    result = subprocess.run(
        [
            "bash", "scripts/linux/train_celeba_latent.sh",
            "--dry-run", "--mode", "continue",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    jobs = [
        line.removeprefix("[PLAN] job=")
        for line in result.stdout.splitlines()
        if line.startswith("[PLAN] job=") and line != "[PLAN] job=reflow_pairs"
    ]
    assert jobs == [
        "fm", "fm_lognorm", "mf", "mf_hutchinson", "mf_distill",
        "consistency", "reflow",
    ]
    assert result.stdout.count(" train.py --algorithm ") == 7
    assert result.stdout.count(" --mode continue ") == 7
    assert "[PLAN] machine_label=linux-" in result.stdout
    assert "[PLAN] gpu_name=" in result.stdout
    assert "[PLAN] gpu_memory_mb=" in result.stdout
    assert "Dry-run complete; no training, pair generation, or log writes occurred." in result.stdout


def test_reflow_dry_run_can_explicitly_accept_older_teacher_source() -> None:
    result = subprocess.run(
        [
            "bash", "scripts/linux/train_celeba_latent.sh",
            "--dry-run", "--only", "reflow", "--mode", "fresh",
            "--allow-teacher-source-mismatch",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--allow-source-identity-mismatch" in result.stdout
