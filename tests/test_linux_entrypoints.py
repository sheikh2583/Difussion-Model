import json
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts import interactive_train


PROJECT_ROOT = Path(__file__).resolve().parents[1]
POSIX_ENTRYPOINTS = (
    PROJECT_ROOT / "scripts" / "linux" / "init.sh",
    PROJECT_ROOT / "scripts" / "linux" / "train.sh",
    PROJECT_ROOT / "scripts" / "linux" / "setup.sh",
)


def test_posix_entrypoints_have_lf_shebangs():
    for path in POSIX_ENTRYPOINTS:
        content = path.read_bytes()
        assert content.startswith(b"#!/usr/bin/env sh\n")
        assert b"\r\n" not in content


def test_posix_entrypoints_parse_with_system_sh():
    shell = shutil.which("sh")
    if shell is None:
        pytest.skip("A POSIX shell is not installed on this host")
    for path in POSIX_ENTRYPOINTS:
        subprocess.run([shell, "-n", str(path)], check=True)


def test_main_launcher_dispatches_all_latent_suite():
    text = (PROJECT_ROOT / "scripts/linux/train.sh").read_text(encoding="utf-8")
    assert '"${1:-}" = "--all-latent"' in text
    assert "scripts/linux/train_celeba_latent.sh --only all" in text


def test_initializer_does_not_depend_on_nested_executable_bit():
    text = (PROJECT_ROOT / "scripts" / "linux" / "init.sh").read_text(encoding="utf-8")
    assert 'exec sh "$SCRIPT_DIR/setup.sh"' in text


def test_setup_installs_venv_support_on_minimal_linux():
    text = (PROJECT_ROOT / "scripts" / "linux" / "setup.sh").read_text(encoding="utf-8")
    assert "python_can_create_venv" in text
    assert "python3-venv" in text
    assert "import ensurepip, venv" in text


def test_training_menu_lists_choices_without_importing_model_stack():
    result = subprocess.run(
        [sys.executable, "scripts/interactive_train.py", "--list"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "cifar10:fm" in result.stdout
    assert "celeba:reflow" in result.stdout
    assert "celeba_latent:fm" in result.stdout
    assert "celeba_latent:fm_lognorm" in result.stdout
    assert "celeba_latent:mf" in result.stdout
    assert "celeba_latent:mf_hutchinson" in result.stdout
    assert "celeba_latent:mf_distill" in result.stdout
    assert "celeba_latent:consistency" in result.stdout
    assert "celeba_latent:reflow" in result.stdout


def test_latent_prerequisites_use_latent_fm_and_pair_generator(monkeypatch):
    choice = interactive_train.CHOICE_BY_KEY["celeba_latent:reflow"]
    commands = []

    monkeypatch.setattr(interactive_train.sys, "platform", "linux")
    monkeypatch.setattr(
        interactive_train,
        "run",
        lambda command, dry_run: commands.append(command),
    )

    interactive_train.train(
        choice.algorithm, choice.config, dry_run=True, mode="continue"
    )

    assert commands[0] == [
        "bash", "scripts/linux/train_celeba_latent.sh",
        "--only", "reflow", "--mode", "continue",
    ]


def test_latent_menu_uses_windows_launcher_on_windows(monkeypatch):
    commands = []
    monkeypatch.setattr(interactive_train.sys, "platform", "win32")
    monkeypatch.setattr(
        interactive_train,
        "run",
        lambda command, dry_run: commands.append(command),
    )

    interactive_train.train(
        "mf_hutchinson",
        "config/mf_hutchinson_celeba_latent.json",
        dry_run=True,
        mode="fresh",
    )

    assert commands[0] == [
        "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
        "-File", "scripts/windows/train_celeba_latent.ps1",
        "-Only", "mf_hutchinson", "-Mode", "fresh",
    ]


def test_teacher_lookup_accepts_numbered_checkpoint_layout(tmp_path, monkeypatch):
    config_dir = tmp_path / "config"
    checkpoint = (
        tmp_path
        / "results"
        / "fm_cifar10"
        / "checkpoints"
        / "run_1"
        / "FlowMatchingAlgorithm_epoch100.pt"
    )
    config_dir.mkdir()
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint-placeholder")
    (config_dir / "consistency.json").write_text(
        json.dumps(
            {
                "dataset": {"name": "cifar10"},
                "algorithm_kwargs": {
                    "teacher_checkpoint": (
                        "results/fm_cifar10/checkpoints/"
                        "FlowMatchingAlgorithm_epoch100.pt"
                    )
                },
            }
        ),
        encoding="utf-8",
    )
    choice = interactive_train.TrainingChoice(
        "test", "Test", "consistency", "config/consistency.json", "FM teacher"
    )
    monkeypatch.setattr(interactive_train, "PROJECT_ROOT", tmp_path)

    with patch.object(
        interactive_train,
        "train",
        side_effect=AssertionError("existing numbered checkpoint must be reused"),
    ):
        resolved = interactive_train.ensure_teacher(choice, dry_run=False)

    assert resolved == checkpoint
