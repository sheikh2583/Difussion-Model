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
        "train_celeba_latent.sh",
        "make_summary.sh",
        "make_thesis_context.sh",
        "refresh_thesis_context.sh",
        "generate_cifar10_outputs.sh",
        "generate_celeba_outputs.sh",
    ):
        path = LINUX_DIR / name
        assert path.stat().st_mode & 0o111
        interpreter = bash if name in (
            "train_cifar.sh",
            "train_all_datasets.sh",
            "train_celeba_latent.sh",
            "make_summary.sh",
            "make_thesis_context.sh",
            "refresh_thesis_context.sh",
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


def test_linux_cifar_launcher_uses_v3_meanflow_and_does_not_mutate_git() -> None:
    text = (LINUX_DIR / "train_all_datasets.sh").read_text(encoding="utf-8")
    assert "config/mf_v3_exact_jvp_b128.json" in text
    assert "training_logs/" in text
    assert 'DEVICE_LOG_DIR="training_logs/$GPU_TOKEN"' in text
    assert 'PIXEL_LOG_DIR="$DEVICE_LOG_DIR/pixel"' in text
    assert "representation_space=pixel" in text
    assert "nvidia-smi --query-gpu=name" in text
    assert "nvidia-smi --query-gpu=memory.total" in text
    assert 'DATASETS=(cifar10 celeba)' in text
    assert 'ALGORITHMS=(fm fm_lognorm mf mf_distill consistency reflow)' in text
    assert '--cifar-backbone "$CIFAR_BACKBONE"' in text
    assert "config/cifar_legacy/fm.json" in (
        LINUX_DIR / "train_all.sh"
    ).read_text(encoding="utf-8")
    for command in ("git add", "git commit", "git push"):
        assert command not in text


def test_windows_cifar_launcher_selects_legacy_backbone_without_forcing_mode() -> None:
    suite = (WINDOWS_DIR / "train_all.ps1").read_text(encoding="utf-8")
    wrapper = (WINDOWS_DIR / "train_cifar.cmd").read_text(encoding="utf-8")

    assert '$CifarBackbone = "current"' in suite
    assert "config/cifar_legacy/fm.json" in suite
    assert "-Mode continue" not in wrapper


def test_linux_latent_launcher_covers_suite_and_only_stages_logs() -> None:
    text = (LINUX_DIR / "train_celeba_latent.sh").read_text(encoding="utf-8")
    assert 'ALGORITHMS=(fm fm_lognorm mf mf_distill consistency reflow)' in text
    assert 'LOG_DIR="training_logs/$GPU_TOKEN/latent"' in text
    assert 'job_log_dir="$LOG_DIR/$job_name"' in text
    assert "representation_space=latent" in text
    assert "scripts/generate_reflow_pairs_latent.py" in text
    assert "scripts/write_training_log_metadata.py" in text
    assert 'git add -- "$log_path" "$sidecar_path"' in text
    for command in ("git commit", "git push"):
        assert command not in text


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
    assert "scripts/package_thesis_context.py" in summary
    assert "--verify-inputs" in summary
    assert "pgrep" in summary
    assert "--allow-running" in summary
    assert "--dry-run" in summary
    assert "kill" not in summary

    context = (LINUX_DIR / "make_thesis_context.sh").read_text(encoding="utf-8")
    assert "make_summary.sh" in context
    assert "scripts/package_thesis_context.py" in context
    assert "--skip-summary" in context

    refresh = (LINUX_DIR / "refresh_thesis_context.sh").read_text(encoding="utf-8")
    assert "make_thesis_context.sh" in refresh
    assert "flock" in refresh
    assert "--interval" in refresh
    assert "kill" not in refresh

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

    celeba = (LINUX_DIR / "generate_celeba_outputs.sh").read_text(encoding="utf-8")
    assert "--dataset celeba_latent" in celeba
    assert "scripts/generate_checkpoint_samples.py" in celeba
    assert "--dataset-family celeba" in celeba


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

    celeba = (WINDOWS_DIR / "generate_celeba_outputs.ps1").read_text(encoding="utf-8")
    assert '"celeba_latent"' in celeba
    assert "scripts/generate_checkpoint_samples.py" in celeba

    windows_summary = (WINDOWS_DIR / "make_summary.ps1").read_text(encoding="utf-8")
    assert "scripts/package_thesis_context.py" in windows_summary
    assert '"--verify-inputs"' in windows_summary
    windows_context = (WINDOWS_DIR / "make_thesis_context.ps1").read_text(encoding="utf-8")
    assert "make_summary.ps1" in windows_context
    assert "SkipSummary" in windows_context
