"""
CLI entry point: train any registered algorithm on any configured dataset.

Usage
-----
    python train.py --algorithm fm
    python train.py --algorithm fm_lognorm --config config/fm_lognorm_full.json
    python train.py --algorithm mf         --config config/mf_full.json
    python train.py --algorithm mf_distill --config config/mf_distill_full.json
    python train.py --algorithm consistency --config config/consistency_full.json
    python train.py --algorithm reflow     --config config/reflow_full.json

Output directory is derived automatically: results/<experiment_name>_<dataset.name>/
So switching datasets in the config automatically routes to a new directory.
"""
import argparse
from pathlib import Path

from algorithms import ALGORITHM_REGISTRY
from config.config import ExperimentConfig
from experiments.runner import ExperimentRunner
from utils.run_lifecycle import (
    archive_existing_run,
    has_run_artifacts,
    run_directory,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a generative-model algorithm.")
    parser.add_argument(
        "--algorithm",
        choices=list(ALGORITHM_REGISTRY.keys()),
        required=True,
        help=("Registered algorithm key, including fm, fm_lognorm, mf, "
              "mf_distill, consistency, reflow, or mock."),
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to a JSON config preset (e.g. config/mf_full.json). "
             "If omitted, uses ExperimentConfig defaults.",
    )
    parser.add_argument("--experiment-name", type=str, default=None,
                        help="Override experiment_name from config.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epochs from config.")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size from config (useful for low-memory runs).")
    parser.add_argument(
        "--train-only", action="store_true",
        help=("Disable FID cache preparation, periodic evaluation, and final "
              "sampling. Intended for a short user-operated diagnostic probe."),
    )
    start_group = parser.add_mutually_exclusive_group()
    start_group.add_argument(
        "--mode",
        choices=("continue", "fresh"),
        help=("Run lifecycle: 'continue' resumes the latest checkpoint; 'fresh' "
              "archives any existing run before starting at epoch 1."),
    )
    start_group.add_argument(
        "--resume", type=str, default=None,
        help=("Legacy/advanced resume option: checkpoint path or 'auto'. "
              "Prefer --mode continue for normal use."),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg  = ExperimentConfig.load(args.config) if args.config else ExperimentConfig()
    if args.experiment_name:
        cfg.experiment_name = args.experiment_name
    if args.epochs is not None:
        if args.epochs < 1:
            raise ValueError("--epochs must be at least 1")
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        if args.batch_size < 1:
            raise ValueError("--batch-size must be at least 1")
        cfg.batch_size = args.batch_size

    project_root = Path(__file__).resolve().parent
    canonical_run_dir = run_directory(cfg, project_root)
    existing = has_run_artifacts(canonical_run_dir)

    resume_checkpoint = args.resume
    if args.mode == "fresh":
        archived = archive_existing_run(canonical_run_dir)
        if archived is not None:
            print(f"[fresh] Previous run preserved at: {archived}")
        else:
            print("[fresh] No previous run artifacts found; starting at epoch 1.")
    elif args.mode == "continue":
        if existing:
            resume_checkpoint = "auto"
            print(f"[continue] Resuming the latest checkpoint in: {canonical_run_dir}")
        else:
            resume_checkpoint = None
            print("[continue] No previous run found; starting the first run at epoch 1.")
    elif resume_checkpoint is None and existing:
        raise SystemExit(
            f"Run artifacts already exist at {canonical_run_dir}. "
            "Choose --mode continue to resume or --mode fresh to preserve them "
            "in results/history and restart from epoch 1."
        )

    if resume_checkpoint and resume_checkpoint != "auto":
        requested_checkpoint = Path(resume_checkpoint)
        if not requested_checkpoint.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {requested_checkpoint}")

    algorithm_cls = ALGORITHM_REGISTRY[args.algorithm]
    runner = ExperimentRunner(cfg, algorithm_cls, algorithm_key=args.algorithm)
    if args.train_only:
        runner.run_train_only(resume_checkpoint=resume_checkpoint)
    else:
        runner.run_full(resume_checkpoint=resume_checkpoint)


if __name__ == "__main__":
    main()
