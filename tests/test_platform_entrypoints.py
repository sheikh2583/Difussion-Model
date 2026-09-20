from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LINUX_DIR = PROJECT_ROOT / "platform" / "linux"
WINDOWS_DIR = PROJECT_ROOT / "platform" / "windows"


def test_linux_platform_scripts_parse_and_are_executable() -> None:
    shell = shutil.which("sh")
    bash = shutil.which("bash")
    if shell is None or bash is None:
        pytest.skip("POSIX shells are unavailable")

    for name in (
        "init.sh",
        "train.sh",
        "train_cifar.sh",
        "train_all_datasets.sh",
        "make_summary.sh",
        "make_minimal_zip.sh",
    ):
        path = LINUX_DIR / name
        assert path.stat().st_mode & 0o111
        interpreter = bash if name in (
            "train_cifar.sh",
            "train_all_datasets.sh",
            "make_summary.sh",
            "make_minimal_zip.sh",
        ) else shell
        subprocess.run([interpreter, "-n", str(path)], check=True)


def test_platform_wrappers_reference_preserved_entrypoints() -> None:
    assert "init_all.sh" in (LINUX_DIR / "init.sh").read_text(encoding="utf-8")
    assert "train_interactive.sh" in (LINUX_DIR / "train.sh").read_text(
        encoding="utf-8"
    )
    assert "train_all_datasets.sh" in (LINUX_DIR / "train_cifar.sh").read_text(
        encoding="utf-8"
    )
    assert "INIT_ALL.cmd" in (WINDOWS_DIR / "init.cmd").read_text(encoding="utf-8")
    assert "TRAIN.cmd" in (WINDOWS_DIR / "train.cmd").read_text(encoding="utf-8")
    assert "scripts\\train_all.ps1" in (
        WINDOWS_DIR / "train_cifar.cmd"
    ).read_text(encoding="utf-8")


def test_linux_cifar_launcher_uses_v3_meanflow_and_tracks_only_text_log() -> None:
    text = (LINUX_DIR / "train_all_datasets.sh").read_text(encoding="utf-8")
    assert "config/mf_v3_exact_jvp_b128.json" in text
    assert "training_logs/" in text
    assert 'DATASETS=(cifar10 celeba)' in text
    assert 'ALGORITHMS=(fm fm_lognorm mf mf_distill consistency reflow)' in text
    assert 'git add -- "${LOG_FILES[@]}"' in text
    assert 'git commit -m "logs: record unattended dataset training' in text


def test_generated_outputs_remain_ignored() -> None:
    result = subprocess.run(
        [
            "git",
            "check-ignore",
            "results/example/checkpoints/run_1/model.pt",
            "data/raw/cifar-10-python.tar.gz",
            "data/reflow_pairs_cifar10.pt",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    ignored = set(result.stdout.splitlines())
    assert "results/example/checkpoints/run_1/model.pt" in ignored
    assert "data/raw/cifar-10-python.tar.gz" in ignored
    assert "data/reflow_pairs_cifar10.pt" in ignored


def test_linux_reporting_helpers_are_training_safe() -> None:
    summary = (LINUX_DIR / "make_summary.sh").read_text(encoding="utf-8")
    package = (LINUX_DIR / "make_minimal_zip.sh").read_text(encoding="utf-8")

    assert "scripts/aggregate_results.py" in summary
    assert "scripts/package_review.py" in package
    for text in (summary, package):
        assert "pgrep" in text
        assert "--allow-running" in text
        assert "--dry-run" in text
        assert "kill" not in text
