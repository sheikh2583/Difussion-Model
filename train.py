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
import hashlib
import os
from pathlib import Path

from algorithms import ALGORITHM_REGISTRY
from config.config import ExperimentConfig
from experiments.runner import ExperimentRunner
from utils.run_lifecycle import (
    archive_existing_run,
    has_run_artifacts,
    run_directory,
)
from utils.checkpoint_runs import (
    checkpoint_run_number_from_path,
    latest_epoch_checkpoint,
    latest_checkpoint_run_number,
    migrate_legacy_checkpoint_layout,
    next_checkpoint_run_number,
)
from utils.run_environment import add_config_hash, collect_run_environment
from utils.checkpoint_provenance import build_provenance, validate_checkpoint_file
from utils.gpu_lock import acquire_gpu_lock
from utils.algorithm_compatibility import validate_algorithm_dataset


def verified_completed_checkpoint(
    run_dir: Path,
    *,
    algorithm_cls,
    algorithm_key: str,
    cfg: ExperimentConfig,
    run_number: int,
) -> Path | None:
    """Return the target checkpoint only after its provenance is validated."""
    latest = latest_epoch_checkpoint(
        run_dir,
        class_name=algorithm_cls.__name__,
        maximum_epoch=cfg.epochs,
        run_number=run_number,
    )
    if latest is None or latest[0] != cfg.epochs:
        return None
    expected = build_provenance(cfg, algorithm_cls, algorithm_key)
    if not validate_checkpoint_file(latest[1], expected):
        raise SystemExit(
            f"Cannot skip completed legacy checkpoint without verifiable provenance: "
            f"{latest[1]}. Preserve it and use --mode fresh for a new run, or "
            "evaluate it explicitly with evaluate.py."
        )
    return latest[1]


def print_startup_summary(
    *,
    config_path: str | None,
    algorithm_key: str,
    cfg: ExperimentConfig,
    result_dir: Path,
    run_environment: dict,
) -> None:
    """Print the complete run identity before device or dataset construction."""
    kwargs = cfg.algorithm_kwargs
    resolved_config = (
        str(Path(config_path).expanduser().resolve())
        if config_path
        else "<ExperimentConfig defaults>"
    )
    lines = [
        ("config", resolved_config),
        ("algorithm", algorithm_key),
        ("experiment", cfg.experiment_name),
        ("backbone", cfg.backbone.name),
        ("result_dir", str(result_dir.resolve())),
        ("batch_size", cfg.batch_size),
        ("epochs", cfg.epochs),
        ("seed", cfg.seed),
        ("learning_rate", cfg.optim.learning_rate),
        ("scheduler", cfg.optim.scheduler),
        ("gradient_clip_norm", cfg.optim.gradient_clip_norm),
    ]
    if algorithm_key == "mf":
        lines.extend([
            ("use_exact_jvp", kwargs.get("use_exact_jvp", False)),
            ("fd_force_fp32", kwargs.get("fd_force_fp32", False)),
            ("p_same", kwargs.get("p_same", 0.25)),
            ("p_fd_step", kwargs.get("p_fd_step", 0.5)),
            (
                "jvp_delta_range",
                f"{kwargs.get('jvp_delta_start', 1e-2)} -> "
                f"{kwargs.get('jvp_delta_end', 1e-4)}",
            ),
        ])
    elif algorithm_key == "mf_hutchinson":
        lines.extend([
            ("p_same", kwargs.get("p_same", 0.1)),
            ("p_hutchinson_step", kwargs.get("p_hutchinson_step", 0.8)),
            ("n_probes", kwargs.get("n_probes", 1)),
        ])
    lines.extend([
        ("machine_label", run_environment.get("machine_label")),
        ("code_identity", run_environment.get("code_identity")),
        ("source_identity_sha256", run_environment.get("source_identity_sha256")),
        ("selected_config_file_sha256", run_environment.get("selected_config_file_sha256")),
        ("lifecycle_mode", run_environment.get("lifecycle_mode")),
        ("checkpoint_series", run_environment.get("checkpoint_series")),
        ("parent_suite_timestamp", run_environment.get("parent_suite_timestamp")),
    ])
    print("[startup] Resolved experiment configuration (before dataset loading):")
    for key, value in lines:
        print(f"[startup] {key}: {value}")


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
        "--checkpoint-every", type=int, default=None,
        help="Save a resumable .pt file and self-contained ZIP every N epochs.",
    )
    parser.add_argument(
        "--machine-label", default=None,
        help=("Human-readable machine name stored in manifests and every metric "
              "row. Defaults to DIFFUSION_MACHINE_LABEL or the hostname."),
    )
    parser.add_argument(
        "--train-only", action="store_true",
        help=("Disable FID cache preparation, periodic evaluation, and final "
              "sampling. Intended for a short user-operated diagnostic probe."),
    )
    parser.add_argument(
        "--evaluate-only", action="store_true",
        help="Run evaluation explicitly from the selected final checkpoint; do not train.",
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


def _main():
    args = parse_args()
    if args.evaluate_only and args.train_only:
        raise SystemExit("--evaluate-only and --train-only are mutually exclusive")
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
    if args.checkpoint_every is not None:
        if args.checkpoint_every < 1:
            raise ValueError("--checkpoint-every must be at least 1")
        cfg.checkpoint_frequency_epochs = args.checkpoint_every

    # Enforce representation-specific algorithms before lifecycle handling can
    # archive or otherwise touch an existing result directory.
    validate_algorithm_dataset(args.algorithm, cfg)

    project_root = Path(__file__).resolve().parent
    canonical_run_dir = run_directory(cfg, project_root)
    migrated = migrate_legacy_checkpoint_layout(canonical_run_dir)
    if migrated:
        print(
            f"[checkpoints] Organized {len(migrated)} legacy files under "
            "checkpoints/run_1/."
        )
    existing = has_run_artifacts(canonical_run_dir)

    resume_checkpoint = args.resume
    if args.mode == "fresh":
        archived = archive_existing_run(
            canonical_run_dir, preserve_checkpoints=True
        )
        if archived is not None:
            print(f"[fresh] Previous run preserved at: {archived}")
        else:
            print("[fresh] No previous run artifacts found; starting at epoch 1.")
    elif args.mode == "continue":
        latest_saved_checkpoint = latest_epoch_checkpoint(canonical_run_dir)
        if latest_saved_checkpoint is not None:
            resume_checkpoint = "auto"
            print(f"[continue] Resuming the latest checkpoint in: {canonical_run_dir}")
        elif existing:
            # A stop before the first scheduled checkpoint may leave config and
            # log files but no resumable state. Preserve that diagnostic evidence
            # and begin a clean epoch-1 attempt instead of failing or mixing logs.
            archived = archive_existing_run(
                canonical_run_dir, preserve_checkpoints=True
            )
            existing = False
            resume_checkpoint = None
            print(
                "[continue] Existing artifacts contain no checkpoint; preserved "
                f"them at {archived} and starting a clean run at epoch 1."
            )
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

    explicit_run_number = (
        checkpoint_run_number_from_path(Path(args.resume))
        if args.resume and args.resume != "auto" else None
    )
    if explicit_run_number is not None:
        checkpoint_run_number = explicit_run_number
    elif (args.mode == "continue" or args.resume) and existing:
        checkpoint_run_number = latest_checkpoint_run_number(canonical_run_dir)
        if checkpoint_run_number is None:
            raise SystemExit(
                f"Cannot continue {canonical_run_dir}: no numbered checkpoint run exists."
            )
    else:
        checkpoint_run_number = next_checkpoint_run_number(canonical_run_dir)
    print(f"[checkpoints] Active checkpoint series: run_{checkpoint_run_number}")

    algorithm_cls = ALGORITHM_REGISTRY[args.algorithm]
    completed_checkpoint = (
        verified_completed_checkpoint(
            canonical_run_dir,
            algorithm_cls=algorithm_cls,
            algorithm_key=args.algorithm,
            cfg=cfg,
            run_number=checkpoint_run_number,
        )
        if args.mode == "continue" else None
    )
    if completed_checkpoint is not None:
        if not args.evaluate_only:
            print(
                f"[continue] Verified compatible completed checkpoint; skipping "
                f"training and implicit evaluation: {completed_checkpoint}"
            )
            return

    # Collect this once and pass the same authoritative record into the runner.
    # This happens before ExperimentRunner resolves CUDA or constructs a dataset.
    run_environment = add_config_hash(
        collect_run_environment(project_root, machine_label=args.machine_label), cfg
    )
    run_environment.update({
        "lifecycle_mode": args.mode or ("resume" if args.resume else "unspecified"),
        "checkpoint_series": f"run_{checkpoint_run_number}",
    })
    if args.config:
        config_bytes = Path(args.config).expanduser().resolve().read_bytes()
        run_environment["selected_config_file_sha256"] = hashlib.sha256(config_bytes).hexdigest()
    print_startup_summary(
        config_path=args.config,
        algorithm_key=args.algorithm,
        cfg=cfg,
        result_dir=canonical_run_dir,
        run_environment=run_environment,
    )

    runner = ExperimentRunner(
        cfg,
        algorithm_cls,
        algorithm_key=args.algorithm,
        checkpoint_run_number=checkpoint_run_number,
        machine_label=args.machine_label,
        run_environment=run_environment,
    )
    if args.evaluate_only:
        target = completed_checkpoint
        if target is None:
            raise FileNotFoundError(
                f"No epoch-{cfg.epochs} checkpoint is available for explicit evaluation"
            )
        runner.evaluate_only(str(target))
    elif args.train_only:
        runner.run_train_only(resume_checkpoint=resume_checkpoint)
    else:
        runner.run_full(resume_checkpoint=resume_checkpoint)


def main():
    with acquire_gpu_lock(command="train.py"):
        _main()


if __name__ == "__main__":
    main()
