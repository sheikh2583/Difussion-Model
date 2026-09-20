from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LINUX_DIR = PROJECT_ROOT / "scripts" / "linux"
WINDOWS_DIR = PROJECT_ROOT / "scripts" / "windows"


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
        "generate_cifar10_outputs.sh",
        "generate_celeba_outputs.sh",
    ):
        path = LINUX_DIR / name
        assert path.stat().st_mode & 0o111
        interpreter = bash if name in (
            "train_cifar.sh",
            "train_all_datasets.sh",
            "make_summary.sh",
            "generate_cifar10_outputs.sh",
            "generate_celeba_outputs.sh",
        ) else shell
        subprocess.run([interpreter, "-n", str(path)], check=True)


def test_platform_wrappers_reference_shared_entrypoints() -> None:
    assert "setup.sh" in (LINUX_DIR / "init.sh").read_text(encoding="utf-8")
    assert "scripts/interactive_train.py" in (LINUX_DIR / "train.sh").read_text(encoding="utf-8")
    assert "train_all_datasets.sh" in (LINUX_DIR / "train_cifar.sh").read_text(
        encoding="utf-8"
    )
    assert "setup.ps1" in (WINDOWS_DIR / "init.cmd").read_text(encoding="utf-8")
    assert "scripts\\interactive_train.py" in (WINDOWS_DIR / "train.cmd").read_text(encoding="utf-8")
    assert "scripts\\windows\\train_all.ps1" in (
        WINDOWS_DIR / "train_cifar.cmd"
    ).read_text(encoding="utf-8")


def test_linux_cifar_launcher_uses_v3_meanflow_and_tracks_only_text_log() -> None:
    text = (LINUX_DIR / "train_all_datasets.sh").read_text(encoding="utf-8")
    assert "config/mf_v3_exact_jvp_b128.json" in text
    assert "training_logs/" in text
    assert 'DEVICE_LOG_DIR="training_logs/$GPU_TOKEN"' in text
    assert "nvidia-smi --query-gpu=name" in text
    assert "nvidia-smi --query-gpu=memory.total" in text
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

    assert "scripts/aggregate_results.py" in summary
    assert "pgrep" in summary
    assert "--allow-running" in summary
    assert "--dry-run" in summary
    assert "kill" not in summary

    for name, dataset in (
        ("generate_cifar10_outputs.sh", "cifar10"),
        ("generate_celeba_outputs.sh", "celeba"),
    ):
        text = (LINUX_DIR / name).read_text(encoding="utf-8")
        assert "scripts/generate_result_gifs.py" in text
        assert f"--dataset {dataset}" in text
        assert "pgrep" in text
        assert "--allow-running" in text
        assert "--dry-run" in text
        assert "kill" not in text


def test_windows_reporting_helpers_are_training_safe() -> None:
    expected = {
        "make_summary.ps1": "scripts/aggregate_results.py",
        "generate_cifar10_outputs.ps1": '"cifar10"',
        "generate_celeba_outputs.ps1": '"celeba"',
    }
    for name, command in expected.items():
        text = (WINDOWS_DIR / name).read_text(encoding="utf-8")
        assert command in text
        assert "Get-CimInstance Win32_Process" in text
        assert "AllowRunning" in text
        assert "DryRun" in text
        assert "Stop-Process" not in text
