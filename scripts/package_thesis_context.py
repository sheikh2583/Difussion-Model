#!/usr/bin/env python3
"""Build a compact, external-agent-friendly thesis context ZIP.

Includes source, configs, tests, report material, metrics, logs, provenance,
plots, and sample images. Excludes datasets, environments, raw checkpoint
tensors, checkpoint archives, and global FID caches.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

try:
    from scripts.verify_project_layout import active_training_processes
except ModuleNotFoundError:  # Direct execution: sys.path starts at scripts/.
    from verify_project_layout import active_training_processes


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "thesis_context.zip"
EXCLUDED_PARTS = {".git", "venv", "__pycache__"}
EXCLUDED_SUFFIXES = {".pt", ".zip", ".npz"}
EXPLICIT_IGNORED_CONTEXT_PATHS = (
    "AGENTS.md",
    "ANTIGRAVITY_PRETRAINED_LATENT_PLAN.md",
    "BLOCKER_DECISIONS.md",
    "CODEX_SELF_TRAINED_CODEC_PLAN.md",
    "CROSS_TRACK.md",
)
REQUIRED_SOURCE_PATHS = (
    "README.md",
    "THESIS_SUMMARY.md",
    "docs/COMPARISON_PROTOCOL.md",
    "scripts/aggregate_results.py",
    "scripts/generate_checkpoint_samples.py",
    "scripts/generate_result_gifs.py",
    "scripts/package_thesis_context.py",
    "scripts/preflight_mf_v2.py",
    "scripts/linux/make_summary.sh",
    "scripts/linux/make_thesis_context.sh",
    "scripts/linux/refresh_thesis_context.sh",
    "web/inference_server.py",
    "web/inference_ui.html",
    "tests/test_aggregate_results.py",
    "tests/test_checkpoint_samples.py",
    "tests/test_result_gifs.py",
    "tests/test_mean_flow_exact_jvp.py",
    "tests/test_web_catalog.py",
)
REQUIRED_AGGREGATE_PATHS = (
    "results/aggregate/combined_metrics.jsonl",
    "results/aggregate/summary.csv",
    "results/aggregate/training_cost_table.csv",
    "results/aggregate/comparison_coverage.csv",
    "results/aggregate/algorithm_progression_vs_fm.csv",
    "results/aggregate/cross_dataset_consistency.csv",
    "results/aggregate/pixel_vs_latent.csv",
    "results/aggregate/quality_compute_pareto.csv",
    "results/aggregate/comparison_manifest.json",
    "results/aggregate/training_log_catalog.csv",
    "results/aggregate/training_log_catalog.json",
    "results/aggregate/TRAINING_LOG_INDEX.md",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--allow-running",
        action="store_true",
        help="Permit a read-only snapshot while training is active.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--verify-inputs",
        action="store_true",
        help="Validate handoff inputs and Git tracking without creating a ZIP.",
    )
    parser.add_argument(
        "--max-mib",
        type=float,
        default=100.0,
        help="Refuse a larger uncompressed context (default: 100 MiB).",
    )
    return parser.parse_args()


def source_files(root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return [
        root / name.decode("utf-8")
        for name in result.stdout.split(b"\0")
        if name and (root / name.decode("utf-8")).is_file()
    ]


def tracked_paths(root: Path) -> set[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"], cwd=root, check=True, capture_output=True
    )
    return {
        name.decode("utf-8") for name in result.stdout.split(b"\0") if name
    }


def git_snapshot(root: Path) -> dict:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=root, check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    return {
        "head": revision,
        "worktree_clean": not status,
        "status": status,
    }


def useful_result_files(root: Path) -> list[Path]:
    results = root / "results"
    selected = []
    if not results.is_dir():
        return selected
    for path in results.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(results)
        if relative.parts and relative.parts[0] in {"exports", "metrics"}:
            continue
        if "checkpoints" in relative.parts or path.suffix.lower() in EXCLUDED_SUFFIXES:
            continue
        selected.append(path)
    return selected


def explicit_ignored_context_files(root: Path) -> list[Path]:
    """Select small ignored provenance records; never select binary payloads."""
    selected = [root / relative for relative in EXPLICIT_IGNORED_CONTEXT_PATHS]
    selected.extend((root / "data/latent_cache").glob("*/manifest.json"))
    selected.extend((root / "data/pretrained").glob("*/README.md"))
    return sorted(path for path in selected if path.is_file())


def context_files(root: Path) -> list[Path]:
    files = set(source_files(root))
    files.update(useful_result_files(root))
    files.update(explicit_ignored_context_files(root))
    logs = root / "training_logs"
    if logs.is_dir():
        files.update(path for path in logs.rglob("*") if path.is_file())
    return sorted(
        path
        for path in files
        if path.is_file()
        and not EXCLUDED_PARTS.intersection(path.relative_to(root).parts)
        and path.suffix.lower() not in EXCLUDED_SUFFIXES
    )


def training_log_files(files: list[Path], root: Path) -> list[str]:
    """Return every packaged training transcript, regardless of trainer type."""
    return sorted(
        path.relative_to(root).as_posix()
        for path in files
        if path.suffix.lower() == ".log"
        and path.relative_to(root).parts[0] in {"results", "training_logs"}
    )


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _csv_rows(path: Path, required_columns: set[str]) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing = required_columns - fields
        if missing:
            raise ValueError(f"{path}: missing CSV columns {sorted(missing)}")
        return sum(1 for _ in reader)


def validate_handoff_inputs(root: Path, require_tracked: bool = True) -> dict:
    """Fail closed when the Claude/thesis handoff would be incomplete or stale."""
    errors: list[str] = []
    required = (*REQUIRED_SOURCE_PATHS, *REQUIRED_AGGREGATE_PATHS)
    for relative in required:
        path = root / relative
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"missing or empty required handoff file: {relative}")

    for relative in EXPLICIT_IGNORED_CONTEXT_PATHS:
        path = root / relative
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"missing or empty ignored context record: {relative}")

    tracked = tracked_paths(root)
    if require_tracked:
        for relative in REQUIRED_SOURCE_PATHS:
            if relative not in tracked:
                errors.append(f"required implementation is not Git-tracked: {relative}")

    canonical_metrics = [
        path
        for path in (root / "results").glob("*/metrics/*.jsonl")
        if path.stem == path.parent.parent.name
    ]
    summary_inputs = canonical_metrics + [
        path for path in (root / "results").glob("*/config.json") if path.is_file()
    ] + [root / "scripts/aggregate_results.py"]
    generated = [
        root / "THESIS_SUMMARY.md",
        *(root / relative for relative in REQUIRED_AGGREGATE_PATHS
          if "training_log" not in relative and not relative.endswith("TRAINING_LOG_INDEX.md")),
    ]
    if summary_inputs and all(path.is_file() for path in generated):
        newest_input = max(path.stat().st_mtime_ns for path in summary_inputs)
        for path in generated:
            if path.stat().st_mtime_ns < newest_input:
                errors.append(
                    f"stale generated handoff file: {path.relative_to(root)} is older "
                    "than a canonical metric, run config, or aggregate implementation"
                )

    catalog_inputs = [root / "scripts/catalog_training_logs.py"] + [
        path
        for base in (root / "results", root / "training_logs")
        if base.is_dir()
        for path in base.rglob("*")
        if path.is_file() and path.suffix.lower() in {".log", ".json"}
        and "aggregate" not in path.parts
    ]
    catalog_outputs = [
        root / "results/aggregate/training_log_catalog.csv",
        root / "results/aggregate/training_log_catalog.json",
        root / "results/aggregate/TRAINING_LOG_INDEX.md",
    ]
    if catalog_inputs and all(path.is_file() for path in catalog_outputs):
        newest_catalog_input = max(path.stat().st_mtime_ns for path in catalog_inputs)
        for path in catalog_outputs:
            if path.stat().st_mtime_ns < newest_catalog_input:
                errors.append(
                    f"stale log catalog: {path.relative_to(root)} is older than "
                    "a catalog input"
                )

    try:
        combined = root / "results/aggregate/combined_metrics.jsonl"
        combined_rows = 0
        with combined.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line.strip():
                    json.loads(line)
                    combined_rows += 1
        if combined_rows == 0:
            errors.append("combined_metrics.jsonl has no records")
    except (OSError, json.JSONDecodeError) as error:
        errors.append(f"invalid combined_metrics.jsonl: {error}")
        combined_rows = 0

    csv_checks = {
        "summary.csv": {"dataset_family", "representation", "method_key", "fid_at_20"},
        "comparison_coverage.csv": {"dataset_family", "representation", "method_key"},
        "algorithm_progression_vs_fm.csv": {
            "method_key", "fid_improvement_percent", "comparison_status"
        },
        "cross_dataset_consistency.csv": {"method_key", "transfer_status"},
        "pixel_vs_latent.csv": {"method_key", "comparison_status"},
        "quality_compute_pareto.csv": {"pareto_fid_vs_nfe"},
        "training_log_catalog.csv": {"path", "representation_space"},
    }
    csv_counts: dict[str, int] = {}
    for filename, columns in csv_checks.items():
        path = root / "results/aggregate" / filename
        try:
            count = _csv_rows(path, columns)
            csv_counts[filename] = count
            if count == 0:
                errors.append(f"aggregate table has no rows: {filename}")
        except (OSError, ValueError) as error:
            errors.append(str(error))

    try:
        comparison = json.loads(
            (root / "results/aggregate/comparison_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        if comparison.get("schema_version") != 1:
            errors.append("comparison_manifest.json has an unsupported schema")
        if not comparison.get("metric_inventory"):
            errors.append("comparison_manifest.json has no metric inventory")
    except (OSError, json.JSONDecodeError) as error:
        errors.append(f"invalid comparison_manifest.json: {error}")

    try:
        catalog = json.loads(
            (root / "results/aggregate/training_log_catalog.json").read_text(
                encoding="utf-8"
            )
        )
        if not catalog:
            errors.append("training_log_catalog.json has no entries")
    except (OSError, json.JSONDecodeError) as error:
        errors.append(f"invalid training_log_catalog.json: {error}")

    if errors:
        raise ValueError("Handoff validation failed:\n- " + "\n- ".join(errors))
    return {
        "required_file_count": len(required),
        "explicit_ignored_context_records": len(
            explicit_ignored_context_files(root)
        ),
        "combined_metric_rows": combined_rows,
        "csv_rows": csv_counts,
        "required_sources_git_tracked": True,
    }


def verify_archive(path: Path, manifest: dict) -> None:
    """Verify ZIP CRCs, required inventory, sizes, and SHA-256 member digests."""
    expected = {entry["path"]: entry for entry in manifest["files"]}
    with zipfile.ZipFile(path) as archive:
        bad_member = archive.testzip()
        if bad_member:
            raise OSError(f"Corrupt archive member: {bad_member}")
        names = set(archive.namelist())
        required = set(REQUIRED_SOURCE_PATHS) | set(REQUIRED_AGGREGATE_PATHS)
        required.update(EXPLICIT_IGNORED_CONTEXT_PATHS)
        missing = required - names
        if missing:
            raise OSError(f"Archive is missing required members: {sorted(missing)}")
        for name, metadata in expected.items():
            payload = archive.read(name)
            if len(payload) != metadata["bytes"]:
                raise OSError(f"Archive member size mismatch: {name}")
            if hashlib.sha256(payload).hexdigest() != metadata["sha256"]:
                raise OSError(f"Archive member digest mismatch: {name}")


def main() -> int:
    args = parse_args()
    root = PROJECT_ROOT
    output = args.output.expanduser().resolve()
    training = active_training_processes(root)
    validation = validate_handoff_inputs(root)
    if args.verify_inputs:
        print(json.dumps(validation, indent=2, sort_keys=True))
        return 0
    files = context_files(root)
    packaged_training_logs = training_log_files(files, root)
    total = sum(path.stat().st_size for path in files)
    if total > args.max_mib * 1024**2:
        raise ValueError(
            f"Context is {total / 1024**2:.1f} MiB uncompressed, exceeding "
            f"--max-mib {args.max_mib:.1f}. Inspect unexpected files before packaging."
        )
    tracked = tracked_paths(root)
    explicitly_included_ignored = {
        path.relative_to(root).as_posix()
        for path in explicit_ignored_context_files(root)
    }
    repository = git_snapshot(root)

    print(f"Files: {len(files)}")
    print(f"Uncompressed size: {total / 1024**2:.1f} MiB")
    print(f"Output: {output}")
    if training:
        print(f"Active training-related processes: {len(training)}")
    if args.dry_run:
        for path in files:
            print(path.relative_to(root).as_posix())
        return 0
    if training and not args.allow_running:
        print(
            "Training is active; the thesis context ZIP was not changed.\n"
            "Run this after training, or add --allow-running for a read-only snapshot.",
            file=os.sys.stderr,
        )
        return 3

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_files = []
    with tempfile.NamedTemporaryFile(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent, delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with zipfile.ZipFile(
            temporary_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as archive:
            for path in files:
                relative = path.relative_to(root).as_posix()
                archive.write(path, relative)
                manifest_files.append(
                    {
                        "path": relative,
                        "bytes": path.stat().st_size,
                        "sha256": digest(path),
                        "git_tracked": relative in tracked,
                        "selection": (
                            "explicit_ignored_provenance"
                            if relative in explicitly_included_ignored
                            else "generated_evidence"
                            if relative.startswith(("results/", "training_logs/"))
                            else "repository_source"
                        ),
                    }
                )
            manifest = {
                "format": "diffusion-project-thesis-context-v2",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "training_active_during_snapshot": bool(training),
                "repository": repository,
                "validation": validation,
                "training_logs": packaged_training_logs,
                "exclusions": [
                    "raw datasets",
                    "virtual environments",
                    "raw .pt checkpoints",
                    "checkpoint/bundle ZIPs",
                    "global FID .npz caches",
                    "pretrained weight binaries and latent tensor caches",
                    "IDE state, Python caches, and nested worktrees",
                ],
                "explicit_ignored_inclusions": sorted(explicitly_included_ignored),
                "files": manifest_files,
            }
            archive.writestr("CONTEXT_MANIFEST.json", json.dumps(manifest, indent=2))
            archive.writestr(
                "CONTEXT_README.md",
                "# Thesis context snapshot\n\n"
                "This compact archive is intended for code review and thesis discussion. "
                "It contains source, configuration, tests, report material, run provenance, "
                "metrics, plots, sample images, and training transcripts. Large model and "
                "dataset binaries are deliberately excluded. See `CONTEXT_MANIFEST.json` "
                "for the complete inventory, Git state, and SHA-256 hashes. Begin with "
                "`README.md`, then `THESIS_SUMMARY.md` and "
                "`docs/COMPARISON_PROTOCOL.md`.\n",
            )
        verify_archive(temporary_path, manifest)
        os.replace(temporary_path, output)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    print(f"Created: {output}")
    print(f"SHA-256: {digest(output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
