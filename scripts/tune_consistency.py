#!/usr/bin/env python3
"""Run a small, isolated Consistency-model stability sweep.

This is intentionally separate from full training. Each variant gets its own
output directory, intermediate evaluation is disabled, and the project-wide
GPU lock prevents overlap with training or Reflow pair generation.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import torch

from algorithms.consistency import ConsistencyAlgorithm
from config.config import ExperimentConfig
from experiments.runner import ExperimentRunner


VARIANTS = {
    "baseline": {
        "ema_decay": 0.999,
        "consistency_weight": 1.0,
        "n_timesteps": 18,
    },
    "lower_weight": {
        "ema_decay": 0.999,
        "consistency_weight": 0.5,
        "n_timesteps": 18,
    },
    "faster_ema": {
        "ema_decay": 0.995,
        "consistency_weight": 1.0,
        "n_timesteps": 18,
    },
    "finer_schedule": {
        "ema_decay": 0.999,
        "consistency_weight": 1.0,
        "n_timesteps": 36,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/consistency_full.json")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--output-root",
        default=None,
        help="Session output directory (default: timestamped results directory).",
    )
    parser.add_argument("--lock-file", default="results/.lock")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and display variants without starting training.",
    )
    return parser.parse_args()


@contextmanager
def gpu_lock(path: Path):
    path = path.resolve()
    token = f"pid={os.getpid()}\ncommand=tune_consistency.py\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(
            f"GPU lock already exists: {path}. Confirm the active job has "
            "finished before removing a stale lock."
        ) from exc
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(token)
        yield
    finally:
        try:
            still_ours = path.read_text(encoding="utf-8") == token
            if still_ours:
                path.unlink()
        except FileNotFoundError:
            pass


def validate_args(args: argparse.Namespace, raw: dict) -> Path:
    if args.epochs < 1:
        raise ValueError(f"--epochs must be >= 1, got {args.epochs}")
    if args.batch_size < 1:
        raise ValueError(f"--batch-size must be >= 1, got {args.batch_size}")
    if raw.get("experiment_name") != "consistency":
        raise ValueError("The base config must use experiment_name='consistency'.")
    teacher = raw.get("algorithm_kwargs", {}).get("teacher_checkpoint")
    if not teacher:
        raise ValueError("The base config does not define a teacher_checkpoint.")
    teacher_path = (PROJECT_ROOT / teacher).resolve()
    if not teacher_path.is_file():
        raise FileNotFoundError(f"Teacher checkpoint not found: {teacher_path}")
    return teacher_path


def build_variant_config(
    raw: dict,
    parameters: dict,
    variant_root: Path,
    epochs: int,
    batch_size: int,
) -> dict:
    variant = copy.deepcopy(raw)
    variant["output_dir"] = str(variant_root)
    variant["epochs"] = epochs
    variant["batch_size"] = batch_size
    variant["checkpoint_frequency_epochs"] = epochs
    variant["algorithm_kwargs"].update(parameters)
    # ExperimentRunner.train() has no final evaluation and this value keeps
    # its epoch hook from evaluating during a short tuning run.
    variant["evaluation"]["eval_frequency_epochs"] = epochs + 1
    return variant


def read_losses(run_dir: Path) -> list[float]:
    metrics_path = run_dir / "metrics" / f"{run_dir.name}.jsonl"
    losses = []
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("record_type") == "train_epoch":
                losses.append(float(record["loss"]))
    if not losses:
        raise RuntimeError(f"No training losses were written to {metrics_path}")
    return losses


def write_summary(session_root: Path, records: list[dict]) -> None:
    successful = [
        row for row in records
        if row.get("status") == "ok" and row.get("finite", False)
    ]
    recommended = min(successful, key=lambda row: row["final_loss"])["variant"] \
        if successful else None
    payload = {
        "recommended_variant": recommended,
        "selection_rule": "lowest finite final epoch loss",
        "variants": records,
    }
    (session_root / "summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    lines = [
        "# Consistency tuning results",
        "",
        "Selection rule: lowest finite final epoch loss.",
        "",
        "| Variant | Status | Initial loss | Final loss | Minimum loss |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in records:
        lines.append(
            f"| {row['variant']} | {row['status']} | "
            f"{row.get('initial_loss', 'n/a')} | {row.get('final_loss', 'n/a')} | "
            f"{row.get('minimum_loss', 'n/a')} |"
        )
    lines.extend(["", f"Recommended variant: `{recommended or 'none'}`", ""])
    (session_root / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    teacher_path = validate_args(args, raw)
    print(f"Base config: {config_path}")
    print(f"Teacher:     {teacher_path}")
    for name, parameters in VARIANTS.items():
        print(f"  {name:14s} {parameters}")
    if args.dry_run:
        print("Dry run complete; no training or result files were created.")
        return 0

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    session_root = Path(
        args.output_root or f"results/consistency_tuning/{timestamp}"
    ).resolve()
    if session_root.exists():
        raise FileExistsError(
            f"Output directory already exists: {session_root}. Choose a new path."
        )
    session_root.mkdir(parents=True)

    records = []
    failures = 0
    with gpu_lock(Path(args.lock_file)):
        for name, parameters in VARIANTS.items():
            print(f"\n=== Consistency tuning variant: {name} ===", flush=True)
            variant_root = session_root / name
            variant_raw = build_variant_config(
                raw, parameters, variant_root, args.epochs, args.batch_size
            )
            generated_config = session_root / "configs" / f"{name}.json"
            generated_config.parent.mkdir(parents=True, exist_ok=True)
            generated_config.write_text(
                json.dumps(variant_raw, indent=2), encoding="utf-8"
            )
            record = {"variant": name, "parameters": parameters}
            runner = None
            try:
                cfg = ExperimentConfig.load(str(generated_config))
                runner = ExperimentRunner(cfg, ConsistencyAlgorithm)
                cfg.save(str(Path(runner.run_dir) / "config.json"))
                runner.train()
                losses = read_losses(Path(runner.run_dir))
                record.update({
                    "status": "ok",
                    "finite": all(torch.isfinite(torch.tensor(losses)).tolist()),
                    "initial_loss": losses[0],
                    "final_loss": losses[-1],
                    "minimum_loss": min(losses),
                    "losses": losses,
                })
            except Exception as exc:
                failures += 1
                record.update({"status": "failed", "error": str(exc)})
            finally:
                if runner is not None:
                    del runner
                records.append(record)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    write_summary(session_root, records)
    print(f"\nSummary: {session_root / 'summary.md'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
