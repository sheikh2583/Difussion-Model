#!/usr/bin/env python3
"""Create portable CIFAR-10 and CelebA checkpoint/results bundles.

Only atomically published checkpoint archives are included. Raw ``.pt`` files
are intentionally omitted because every inner checkpoint ZIP already contains
``checkpoint.pt``, its config, environment metadata, and epoch metadata. The
script never imports torch, opens checkpoint tensors, or controls training.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("cifar10", "celeba")
SOURCE_TOOLS = (
    "scripts/aggregate_results.py",
    "scripts/generate_result_gifs.py",
    "utils/plots.py",
    "utils/results.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Package checkpoint archives, configs, provenance, metrics, and "
            "plot-ready aggregates into one ZIP per dataset."
        )
    )
    parser.add_argument(
        "--dataset", choices=(*DATASETS, "all"), default="all"
    )
    parser.add_argument(
        "--results-root", type=Path, default=PROJECT_ROOT / "results"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "results" / "exports"
    )
    parser.add_argument(
        "--latest-only",
        action="store_true",
        help="Include only the highest-epoch checkpoint archive in each run series.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="List bundle contents and sizes only."
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def config_dataset(config: dict[str, Any]) -> str | None:
    dataset = config.get("dataset")
    if isinstance(dataset, dict):
        value = dataset.get("name")
        return str(value).lower() if value else None
    return None


def experiment_dataset(run_dir: Path) -> str | None:
    configured = config_dataset(load_json(run_dir / "config.json"))
    if configured in DATASETS:
        return configured
    for dataset in DATASETS:
        if run_dir.name.endswith(f"_{dataset}") or run_dir.name.endswith(dataset):
            return dataset
    return None


def experiment_dirs(results_root: Path, dataset: str) -> list[Path]:
    return sorted(
        path
        for path in results_root.iterdir()
        if path.is_dir()
        and path.name not in {"aggregate", "exports", "history", "metrics"}
        and experiment_dataset(path) == dataset
    )


def checkpoint_archives(run_dir: Path, latest_only: bool) -> list[Path]:
    archives = sorted(run_dir.glob("checkpoints/run_*/archive/*.zip"))
    if not latest_only:
        return archives
    latest_by_series: dict[Path, tuple[int, Path]] = {}
    for path in archives:
        digits = "".join(character for character in path.stem.rsplit("epoch", 1)[-1] if character.isdigit())
        epoch = int(digits) if digits else -1
        current = latest_by_series.get(path.parent)
        if current is None or epoch > current[0]:
            latest_by_series[path.parent] = (epoch, path)
    return sorted(item[1] for item in latest_by_series.values())


def valid_jsonl_snapshot(path: Path) -> tuple[str, int]:
    """Return complete JSON rows, ignoring only an append-in-progress tail."""
    lines: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError:
                # ResultsWriter appends one row at a time. Only a final partial
                # line can be an in-progress write; older corruption is fatal.
                if not line.endswith("\n"):
                    break
                raise ValueError(f"Invalid JSON in {path}:{line_number}")
            lines.append(line.rstrip("\n"))
    text = "\n".join(lines)
    return (text + "\n" if text else ""), len(lines)


def filtered_jsonl(path: Path, dataset: str) -> tuple[str, int]:
    selected: list[str] = []
    if not path.is_file():
        return "", 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            experiment = str(row.get("experiment", ""))
            if experiment_dataset_name(experiment) == dataset:
                selected.append(json.dumps(row, sort_keys=True))
    return ("\n".join(selected) + "\n" if selected else ""), len(selected)


def experiment_dataset_name(experiment: str) -> str | None:
    for dataset in DATASETS:
        if experiment.endswith(f"_{dataset}") or experiment.endswith(dataset):
            return dataset
    return None


def filtered_csv(path: Path, dataset: str, experiment_field: str) -> tuple[str, int]:
    if not path.is_file():
        return "", 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [
            row for row in reader
            if experiment_dataset_name(str(row.get(experiment_field, ""))) == dataset
        ]
        fieldnames = reader.fieldnames or []
    if not fieldnames:
        return "", 0
    from io import StringIO

    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue(), len(rows)


def matching_source_configs(dataset: str) -> list[Path]:
    matches = []
    for path in sorted((PROJECT_ROOT / "config").glob("*.json")):
        if config_dataset(load_json(path)) == dataset:
            matches.append(path)
    return matches


def readme(dataset: str) -> str:
    return f"""DiffusionProject {dataset.upper()} portable results bundle

This is a read-only snapshot. It contains:
- validated checkpoint archives under results/<experiment>/checkpoints/;
- per-run configs and run-environment provenance;
- canonical metrics JSONL needed to regenerate plots;
- dataset-filtered combined metrics, summary, and training-cost CSV;
- plotting/aggregation source scripts.

Each inner checkpoint ZIP is self-contained and includes checkpoint.pt,
config.json (when present), environment.json (when present), and meta.json.
Raw .pt files are not duplicated in this outer bundle.

Regenerate GIFs after extraction:
  python scripts/generate_result_gifs.py \\
    --metrics aggregate/combined_metrics_{dataset}.jsonl \\
    --results-root results \\
    --output-dir aggregate/animations_{dataset} \\
    --experiments <space-separated experiment directory names>

Regenerate static aggregate plots when the extracted directory retains the
included results/ layout:
  python scripts/aggregate_results.py --results-root results --output-dir aggregate/rebuilt
"""


def add_file(
    archive: zipfile.ZipFile,
    source: Path,
    arcname: str,
    manifest_files: list[dict[str, Any]],
) -> None:
    archive.write(source, arcname=arcname)
    manifest_files.append({"path": arcname, "bytes": source.stat().st_size})


def add_text(
    archive: zipfile.ZipFile,
    text: str,
    arcname: str,
    manifest_files: list[dict[str, Any]],
) -> None:
    archive.writestr(arcname, text)
    manifest_files.append({"path": arcname, "bytes": len(text.encode("utf-8"))})


def collect_plan(
    results_root: Path, dataset: str, latest_only: bool
) -> tuple[list[Path], list[Path], list[Path]]:
    runs = experiment_dirs(results_root, dataset)
    archives = [
        path for run_dir in runs for path in checkpoint_archives(run_dir, latest_only)
    ]
    configs = matching_source_configs(dataset)
    return runs, archives, configs


def ensure_space(output_dir: Path, required_bytes: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(output_dir).free
    reserve = 1024**3
    if free < required_bytes + reserve:
        raise OSError(
            f"Insufficient free space: need about {(required_bytes + reserve) / 1024**3:.1f} GiB "
            f"including reserve, have {free / 1024**3:.1f} GiB"
        )


def build_bundle(
    results_root: Path,
    output_dir: Path,
    dataset: str,
    latest_only: bool,
    dry_run: bool,
) -> Path:
    runs, archives, configs = collect_plan(results_root, dataset, latest_only)
    if not runs:
        raise ValueError(f"No {dataset} experiment directories found under {results_root}")
    if not archives:
        raise ValueError(f"No published {dataset} checkpoint archives found")

    estimated_bytes = sum(path.stat().st_size for path in archives)
    suffix = "_latest" if latest_only else ""
    output = output_dir / f"{dataset}_checkpoints_plot_data{suffix}.zip"
    print(
        f"[{dataset}] runs={len(runs)} checkpoint_archives={len(archives)} "
        f"estimated={estimated_bytes / 1024**3:.2f} GiB output={output}"
    )
    if dry_run:
        for run_dir in runs:
            count = len(checkpoint_archives(run_dir, latest_only))
            print(f"  {run_dir.name}: {count} checkpoint archive(s)")
        return output

    ensure_space(output_dir, estimated_bytes)
    temporary = output.with_name(f"{output.name}.tmp-{os.getpid()}")
    temporary.unlink(missing_ok=True)
    manifest_files: list[dict[str, Any]] = []
    jsonl_rows: dict[str, int] = {}
    try:
        # Inner checkpoint archives are already compressed; ZIP_STORED avoids
        # wasting CPU/GPU-training throughput trying to compress them again.
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as bundle:
            add_text(bundle, readme(dataset), "README.txt", manifest_files)

            for config in configs:
                add_file(bundle, config, f"config/{config.name}", manifest_files)

            for tool in SOURCE_TOOLS:
                source = PROJECT_ROOT / tool
                if source.is_file():
                    add_file(bundle, source, tool, manifest_files)

            requirements = PROJECT_ROOT / "requirements.txt"
            if requirements.is_file():
                add_file(bundle, requirements, "requirements.txt", manifest_files)

            for run_dir in runs:
                prefix = f"results/{run_dir.name}"
                for name in ("config.json", "run_environment.json", "run_environment.jsonl"):
                    source = run_dir / name
                    if source.is_file():
                        add_file(bundle, source, f"{prefix}/{name}", manifest_files)

                metrics_dir = run_dir / "metrics"
                for source in sorted(metrics_dir.glob("*.jsonl")):
                    text, row_count = valid_jsonl_snapshot(source)
                    arcname = f"{prefix}/metrics/{source.name}"
                    add_text(bundle, text, arcname, manifest_files)
                    jsonl_rows[arcname] = row_count

                planned_run_archives = [
                    source for source in archives
                    if run_dir in source.parents
                ]
                for source in planned_run_archives:
                    relative = source.relative_to(run_dir).as_posix()
                    add_file(bundle, source, f"{prefix}/{relative}", manifest_files)

            aggregate = results_root / "aggregate"
            combined_text, combined_count = filtered_jsonl(
                aggregate / "combined_metrics.jsonl", dataset
            )
            combined_name = f"aggregate/combined_metrics_{dataset}.jsonl"
            add_text(bundle, combined_text, combined_name, manifest_files)
            jsonl_rows[combined_name] = combined_count

            summary_text, summary_count = filtered_csv(
                aggregate / "summary.csv", dataset, "source_experiment"
            )
            add_text(
                bundle,
                summary_text,
                f"aggregate/summary_{dataset}.csv",
                manifest_files,
            )
            cost_text, cost_count = filtered_csv(
                aggregate / "training_cost_table.csv", dataset, "experiment"
            )
            add_text(
                bundle,
                cost_text,
                f"aggregate/training_cost_table_{dataset}.csv",
                manifest_files,
            )

            manifest = {
                "dataset": dataset,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "snapshot_note": (
                    "Training may have been active; only fully published checkpoint "
                    "archives and complete JSONL rows present at snapshot time are included."
                ),
                "raw_pt_files_duplicated": False,
                "latest_only": latest_only,
                "experiments": [path.name for path in runs],
                "checkpoint_archive_count": len(archives),
                "summary_rows": summary_count,
                "training_cost_rows": cost_count,
                "jsonl_rows": jsonl_rows,
                "files": manifest_files,
            }
            bundle.writestr("bundle_manifest.json", json.dumps(manifest, indent=2))

        with zipfile.ZipFile(temporary, "r") as bundle:
            bad_member = bundle.testzip()
            if bad_member is not None:
                raise OSError(f"Corrupt bundle member: {bad_member}")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)

    print(f"Created: {output} ({output.stat().st_size / 1024**3:.2f} GiB)")
    return output


def main() -> int:
    args = parse_args()
    results_root = args.results_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    datasets: Iterable[str] = DATASETS if args.dataset == "all" else (args.dataset,)
    for dataset in datasets:
        build_bundle(
            results_root,
            output_dir,
            dataset,
            latest_only=args.latest_only,
            dry_run=args.dry_run,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
