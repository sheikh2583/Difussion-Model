#!/usr/bin/env python3
"""Beginner-friendly model picker, prerequisite builder, trainer, and packager."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.run_lifecycle import has_run_artifacts, latest_checkpoint


@dataclass(frozen=True)
class TrainingChoice:
    key: str
    label: str
    algorithm: str
    config: str
    dependency: str = ""


CHOICES = (
    TrainingChoice("smoke", "Quick smoke test (recommended first run)", "mock", "config/smoke_fast.json"),
    TrainingChoice("cifar10:fm", "CIFAR-10 - Flow Matching", "fm", "config/fm_full.json"),
    TrainingChoice("cifar10:fm_lognorm", "CIFAR-10 - Flow Matching, logit-normal", "fm_lognorm", "config/fm_lognorm_full.json"),
    TrainingChoice("cifar10:mf", "CIFAR-10 - Mean Flow", "mf", "config/mf_full.json"),
    TrainingChoice("cifar10:mf_distill", "CIFAR-10 - Mean Flow Distillation", "mf_distill", "config/mf_distill_full.json", "FM teacher"),
    TrainingChoice("cifar10:consistency", "CIFAR-10 - Consistency Model", "consistency", "config/consistency_full.json", "FM teacher"),
    TrainingChoice("cifar10:reflow", "CIFAR-10 - Reflow", "reflow", "config/reflow_full.json", "FM teacher + generated pairs"),
    TrainingChoice("celeba:fm", "CelebA 64x64 - Flow Matching", "fm", "config/fm_celeba64.json"),
    TrainingChoice("celeba:fm_lognorm", "CelebA 64x64 - Flow Matching, logit-normal", "fm_lognorm", "config/fm_lognorm_celeba64.json"),
    TrainingChoice("celeba:mf", "CelebA 64x64 - Mean Flow", "mf", "config/mf_celeba64.json"),
    TrainingChoice("celeba:mf_distill", "CelebA 64x64 - Mean Flow Distillation", "mf_distill", "config/mf_distill_celeba64.json", "FM teacher"),
    TrainingChoice("celeba:consistency", "CelebA 64x64 - Consistency Model", "consistency", "config/consistency_celeba64.json", "FM teacher"),
    TrainingChoice("celeba:reflow", "CelebA 64x64 - Reflow", "reflow", "config/reflow_celeba64.json", "FM teacher + generated pairs"),
)
CHOICE_BY_KEY = {choice.key: choice for choice in CHOICES}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--choice", choices=CHOICE_BY_KEY, help="Skip the model menu.")
    parser.add_argument(
        "--mode", choices=("continue", "fresh"),
        help="Resume the canonical run or archive it and start from epoch 1.",
    )
    parser.add_argument(
        "--epochs", type=int,
        help="Total target epoch count (use a value above the latest epoch to extend a run).",
    )
    parser.add_argument("--yes", action="store_true", help="Skip the final confirmation.")
    parser.add_argument("--dry-run", action="store_true", help="Print the complete plan without training.")
    parser.add_argument("--list", action="store_true", help="List accepted --choice values and exit.")
    return parser.parse_args()


def select_choice() -> TrainingChoice:
    print("\nChoose a model to train:\n")
    for index, choice in enumerate(CHOICES, start=1):
        suffix = f" [auto-builds {choice.dependency}]" if choice.dependency else ""
        print(f"  {index:2}. {choice.label}{suffix}")
    while True:
        try:
            answer = input(f"\nEnter 1-{len(CHOICES)}: ").strip()
        except EOFError as error:
            raise SystemExit("No selection received.") from error
        if answer.isdigit() and 1 <= int(answer) <= len(CHOICES):
            return CHOICES[int(answer) - 1]
        print("Please enter one of the displayed numbers.")


def load_config(path: str) -> dict:
    with (PROJECT_ROOT / path).open(encoding="utf-8") as handle:
        return json.load(handle)


def run(command: list[str], dry_run: bool) -> None:
    printable = subprocess.list2cmdline(command)
    print(f"\n> {printable}")
    if not dry_run:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def train(
    algorithm: str,
    config: str,
    dry_run: bool,
    mode: str = "continue",
    epochs: Optional[int] = None,
) -> None:
    command = [
        sys.executable, "train.py", "--algorithm", algorithm,
        "--config", config, "--mode", mode,
    ]
    if epochs is not None:
        command.extend(("--epochs", str(epochs)))
    run(command, dry_run)


def ensure_teacher(choice: TrainingChoice, dry_run: bool) -> Path:
    selected = load_config(choice.config)
    teacher = Path(selected["algorithm_kwargs"]["teacher_checkpoint"])
    if (PROJECT_ROOT / teacher).is_file():
        print(f"[ready] FM teacher: {teacher}")
        return teacher

    dataset = selected["dataset"]["name"]
    teacher_config = "config/fm_celeba64.json" if dataset == "celeba" else "config/fm_full.json"
    print(f"[prerequisite] FM teacher is missing; it will be trained first: {teacher}")
    train("fm", teacher_config, dry_run)
    if not dry_run and not (PROJECT_ROOT / teacher).is_file():
        raise FileNotFoundError(f"FM training completed but expected checkpoint is missing: {teacher}")
    return teacher


def ensure_prerequisites(choice: TrainingChoice, dry_run: bool) -> None:
    if choice.algorithm in {"mf_distill", "consistency"}:
        ensure_teacher(choice, dry_run)
        return
    if choice.algorithm != "reflow":
        return

    selected = load_config(choice.config)
    pairs = Path(selected["algorithm_kwargs"]["pairs_path"])
    if (PROJECT_ROOT / pairs).is_file():
        print(f"[ready] Reflow pairs: {pairs}")
        return

    dataset = selected["dataset"]["name"]
    teacher_choice = CHOICE_BY_KEY[f"{dataset}:mf_distill"]
    teacher = ensure_teacher(teacher_choice, dry_run)
    teacher_config = "config/fm_celeba64.json" if dataset == "celeba" else "config/fm_full.json"
    print(f"[prerequisite] Reflow pairs are missing; 50,000 pairs will be generated: {pairs}")
    run([
        sys.executable,
        "scripts/generate_reflow_pairs.py",
        "--checkpoint", str(teacher),
        "--config", teacher_config,
        "--n-pairs", "50000",
        "--nfe", "50",
        "--output", str(pairs),
    ], dry_run)


def expected_run_dir(choice: TrainingChoice) -> Path:
    config = load_config(choice.config)
    return PROJECT_ROOT / config.get("output_dir", "results") / (
        f"{config['experiment_name']}_{config['dataset']['name']}"
    )


def select_start_mode(
    choice: TrainingChoice,
    requested_mode: Optional[str],
    non_interactive: bool,
) -> str:
    run_dir = expected_run_dir(choice)
    if requested_mode:
        return requested_mode
    if not has_run_artifacts(run_dir):
        print("No existing run was found; a fresh run will be started.")
        return "fresh"
    if non_interactive:
        print("Existing run found; defaulting to continue mode.")
        return "continue"

    checkpoint = latest_checkpoint(run_dir)
    detail = f" (latest checkpoint: epoch {checkpoint[0]})" if checkpoint else ""
    print(f"\nExisting run found at {run_dir}{detail}.")
    print("  1. Continue from the latest checkpoint")
    print("  2. Start fresh (the existing run is moved to results/history)")
    print("  3. Cancel")
    while True:
        answer = input("Choose 1-3: ").strip()
        if answer == "1":
            return "continue"
        if answer == "2":
            return "fresh"
        if answer == "3":
            raise SystemExit("Cancelled.")
        print("Please enter 1, 2, or 3.")


def resolve_target_epochs(
    choice: TrainingChoice,
    mode: str,
    requested_epochs: Optional[int],
    non_interactive: bool,
) -> Optional[int]:
    if requested_epochs is not None:
        if requested_epochs < 1:
            raise SystemExit("--epochs must be at least 1")
        return requested_epochs
    if mode != "continue":
        return None

    checkpoint = latest_checkpoint(expected_run_dir(choice))
    configured_epochs = int(load_config(choice.config).get("epochs", 1))
    if checkpoint is None or checkpoint[0] < configured_epochs:
        return None
    if non_interactive:
        print(
            f"Run already reached epoch {checkpoint[0]}. It will be packaged without "
            "additional training; pass --epochs with a larger total to extend it."
        )
        return None

    default_target = checkpoint[0] + 10
    while True:
        answer = input(
            f"Run is complete at epoch {checkpoint[0]}. New total epoch target "
            f"[{default_target}]: "
        ).strip()
        if not answer:
            return default_target
        if answer.isdigit() and int(answer) > checkpoint[0]:
            return int(answer)
        print(f"Enter an integer greater than {checkpoint[0]}.")


def main() -> int:
    args = parse_args()
    if args.list:
        for choice in CHOICES:
            print(f"{choice.key:24} {choice.label}")
        return 0

    choice = CHOICE_BY_KEY[args.choice] if args.choice else select_choice()
    mode = select_start_mode(
        choice, args.mode, non_interactive=args.yes or args.dry_run
    )
    target_epochs = resolve_target_epochs(
        choice, mode, args.epochs, non_interactive=args.yes or args.dry_run
    )
    print(f"\nSelected: {choice.label}")
    print(f"Run mode: {mode}")
    if target_epochs is not None:
        print(f"Target epochs: {target_epochs}")
    if mode == "fresh" and has_run_artifacts(expected_run_dir(choice)):
        print("The existing run will be preserved under results/history before training.")
    if choice.dependency:
        print(f"Missing prerequisite handling: automatic ({choice.dependency})")
    print("Checkpoints are zipped automatically, and a complete run ZIP is created after training.")

    if not args.yes and not args.dry_run:
        answer = input("Start training? [Y/n]: ").strip().lower()
        if answer in {"n", "no"}:
            print("Cancelled.")
            return 0

    ensure_prerequisites(choice, args.dry_run)
    train(choice.algorithm, choice.config, args.dry_run, mode, target_epochs)

    run_dir = expected_run_dir(choice)
    export = PROJECT_ROOT / "results" / "exports" / f"{run_dir.name}.zip"
    run([
        sys.executable,
        "scripts/package_run.py",
        "--run-dir", str(run_dir),
        "--output", str(export),
    ], args.dry_run)

    if args.dry_run:
        print("\nDry-run complete; no training or files were created.")
    else:
        print(f"\nTraining complete. Portable run archive: {export}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
