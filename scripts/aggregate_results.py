#!/usr/bin/env python3
"""Merge experiment JSONL metrics, build the thesis table, and render plots."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from utils.plots import (
    plot_fid_vs_sampling_time,
    plot_gpu_memory_comparison,
    plot_is_vs_nfe,
    plot_loss_vs_epoch,
    plot_sampling_time_vs_nfe,
    plot_training_time_comparison,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate all experiment metrics into plots and a summary table."
    )
    parser.add_argument("--results-root", default="results")
    parser.add_argument(
        "--output-dir", default="results/aggregate",
        help="Directory for combined JSONL, CSV, and plots.",
    )
    parser.add_argument(
        "--markdown-output",
        default=str(PROJECT_ROOT / "THESIS_SUMMARY.md"),
        help="Path for the concise Markdown thesis summary.",
    )
    return parser.parse_args()


def load_records(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        run_dir = path.parent.parent
        config: dict[str, Any] = {}
        try:
            config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
        dataset_config = config.get("dataset", {})
        dataset = dataset_config.get("name") if isinstance(dataset_config, dict) else None
        if not dataset:
            dataset = next(
                (name for name in ("cifar10", "celeba") if run_dir.name.endswith(f"_{name}")),
                "unknown",
            )
        file_records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON in {path}:{line_number}: {error}"
                    ) from error
                if not isinstance(record, dict):
                    raise ValueError(f"Expected an object in {path}:{line_number}")
                file_records.append(record)

        # Preserve the raw JSONL order long enough to associate periodic
        # sampling/evaluation blocks with their preceding training epoch.
        last_epoch = None
        for record in file_records:
            if record.get("record_type") == "train_epoch" and record.get("epoch"):
                last_epoch = record["epoch"]
            elif (
                record.get("record_type") in {"evaluation", "sampling"}
                and record.get("epoch") is None
            ):
                record["epoch"] = last_epoch

        # Standalone evaluations may not follow a training row. Recover their
        # epoch from the checkpoint filename when it is available.
        for record in file_records:
            if (
                record.get("record_type") == "evaluation"
                and record.get("epoch") is None
            ):
                checkpoint = record.get("checkpoint_path") or ""
                match = re.search(r"epoch(\d+)", checkpoint)
                if match:
                    record["epoch"] = int(match.group(1))

        for record in file_records:
            experiment = path.parent.parent.name
            algorithm_class = record.get("algorithm")

            # Legacy evaluation rows did not record their checkpoint. Add
            # the conventional run_1 path only when that file exists.
            if (
                record.get("record_type") == "evaluation"
                and not record.get("checkpoint_path")
                and record.get("epoch") is not None
                and algorithm_class
            ):
                expected = (
                    path.parent.parent
                    / "checkpoints"
                    / "run_1"
                    / f"{algorithm_class}_epoch{record['epoch']}.pt"
                )
                if expected.is_file():
                    record["checkpoint_path"] = str(expected.resolve())

            machine = record.get("machine_label") or record.get("hostname")
            code_identity = record.get("code_identity") or record.get("git_commit")
            comparison = experiment
            if machine:
                comparison += f"@{machine}"
            if code_identity:
                comparison += f"@{str(code_identity)[:20]}"
            record["algorithm_class"] = algorithm_class
            record["experiment"] = experiment
            record["experiment_name"] = config.get("experiment_name", experiment)
            record["dataset"] = str(dataset).lower()
            record["comparison"] = comparison
            # Plotting utilities group on `algorithm`; use the run name so
            # datasets, machines, and code revisions cannot merge silently.
            record["algorithm"] = comparison
            records.append(record)
    return records


def deduplicate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the latest row for each logical experiment measurement."""
    latest: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in records:
        key = (
            record.get("experiment"),
            record.get("machine_label") or record.get("hostname"),
            record.get("code_identity") or record.get("git_commit"),
            record.get("seed"),
            record.get("record_type"),
            record.get("epoch"),
            record.get("nfe"),
        )
        latest[key] = record
    return sorted(
        latest.values(),
        key=lambda row: (
            str(row.get("algorithm", "")),
            str(row.get("record_type", "")),
            row.get("epoch") if row.get("epoch") is not None else -1,
            row.get("nfe") if row.get("nfe") is not None else -1,
        ),
    )


def build_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_algorithm: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        algorithm = str(record.get("algorithm", "unknown"))
        by_algorithm.setdefault(algorithm, []).append(record)

    rows: list[dict[str, Any]] = []
    for algorithm, algorithm_records in sorted(by_algorithm.items()):
        training = sorted(
            (
                row for row in algorithm_records
                if row.get("record_type") == "train_epoch"
            ),
            key=lambda row: row.get("epoch") or -1,
        )
        evaluations = {
            row.get("nfe"): row
            for row in algorithm_records
            if row.get("record_type") == "evaluation"
        }
        evaluation_rows = [
            row for row in algorithm_records
            if row.get("record_type") == "evaluation"
        ]
        sample_count = next(
            (
                row.get("num_generated_samples")
                for row in reversed(evaluation_rows)
                if row.get("num_generated_samples") is not None
            ),
            "unknown",
        )
        eval_epoch = max(
            (
                row["epoch"] for row in evaluation_rows
                if row.get("epoch") is not None
            ),
            default=None,
        )
        last_training = training[-1] if training else {}
        rows.append({
            "experiment": algorithm,
            "source_experiment": next(
                (row.get("experiment") for row in algorithm_records), None
            ),
            "dataset": next(
                (row.get("dataset") for row in algorithm_records if row.get("dataset")),
                None,
            ),
            "machine_label": next(
                (row.get("machine_label") or row.get("hostname")
                 for row in algorithm_records
                 if row.get("machine_label") or row.get("hostname")),
                None,
            ),
            "git_commit": next(
                (row.get("git_commit") for row in algorithm_records
                 if row.get("git_commit")),
                None,
            ),
            "code_identity": next(
                (row.get("code_identity") for row in algorithm_records
                 if row.get("code_identity")),
                None,
            ),
            "config_sha256": next(
                (row.get("config_sha256") for row in algorithm_records
                 if row.get("config_sha256")),
                None,
            ),
            "gpu_name": next(
                (row.get("gpu_name") for row in algorithm_records
                 if row.get("gpu_name")),
                None,
            ),
            "algorithm_class": next(
                (row.get("algorithm_class") for row in algorithm_records
                 if row.get("algorithm_class")),
                None,
            ),
            "last_epoch": last_training.get("epoch"),
            "backbone_params": last_training.get("parameter_count"),
            "extra_params": last_training.get("algorithm_extra_parameter_count"),
            "training_time_seconds": last_training.get("training_time"),
            "peak_gpu_memory_mb": max(
                (
                    row.get("peak_gpu_memory") or 0
                    for row in training
                ),
                default=None,
            ),
            "eval_epoch": eval_epoch,
            "num_generated_samples": sample_count,
            "fid_at_1": (evaluations.get(1) or {}).get("fid"),
            "fid_at_2": (evaluations.get(2) or {}).get("fid"),
            "fid_at_5": (evaluations.get(5) or {}).get("fid"),
            "fid_at_10": (evaluations.get(10) or {}).get("fid"),
            "fid_at_20": (evaluations.get(20) or {}).get("fid"),
            "fid_at_50": (evaluations.get(50) or {}).get("fid"),
            "fid_at_100": (evaluations.get(100) or {}).get("fid"),
            "is_at_1": (evaluations.get(1) or {}).get("is_mean"),
            "is_at_2": (evaluations.get(2) or {}).get("is_mean"),
            "is_at_5": (evaluations.get(5) or {}).get("is_mean"),
            "is_at_10": (evaluations.get(10) or {}).get("is_mean"),
            "is_at_20": (evaluations.get(20) or {}).get("is_mean"),
            "is_at_50": (evaluations.get(50) or {}).get("is_mean"),
            "is_at_100": (evaluations.get(100) or {}).get("is_mean"),
        })
    return rows


def markdown_value(value: Any, digits: int | None = None) -> str:
    """Format a compact, pipe-safe Markdown table value."""
    if value is None or value == "":
        return "—"
    if isinstance(value, float) and digits is not None:
        return f"{value:.{digits}f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def write_markdown_summary(
    path: Path,
    summary: list[dict[str, Any]],
    training_cost_rows: list[dict[str, Any]],
) -> None:
    """Write a stable, human-readable thesis snapshot from aggregate rows."""
    lines = [
        "# Thesis Results Summary",
        "",
        "Generated from the canonical experiment metric files by "
        "`scripts/aggregate_results.py`. Rerun the generator after active training "
        "finishes to produce the final snapshot.",
        "",
        "## Evaluation metrics",
        "",
        "| Dataset | Experiment | Algorithm | Train epoch | Eval epoch | Samples | "
        "FID@1 | FID@5 | FID@10 | FID@20 | FID@50 | IS@20 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        values = [
            markdown_value(row.get("dataset")),
            markdown_value(row.get("source_experiment")),
            markdown_value(row.get("algorithm_class")),
            markdown_value(row.get("last_epoch")),
            markdown_value(row.get("eval_epoch")),
            markdown_value(row.get("num_generated_samples")),
            markdown_value(row.get("fid_at_1"), 3),
            markdown_value(row.get("fid_at_5"), 3),
            markdown_value(row.get("fid_at_10"), 3),
            markdown_value(row.get("fid_at_20"), 3),
            markdown_value(row.get("fid_at_50"), 3),
            markdown_value(row.get("is_at_20"), 3),
        ]
        lines.append("| " + " | ".join(values) + " |")

    lines.extend([
        "",
        "## Training cost",
        "",
        "| Experiment | Algorithm | Epochs | Training hours | Peak GPU memory (MiB) |",
        "|---|---|---:|---:|---:|",
    ])
    for row in training_cost_rows:
        values = [
            markdown_value(row.get("experiment")),
            markdown_value(row.get("algorithm")),
            markdown_value(row.get("epochs")),
            markdown_value(row.get("total_training_hours"), 2),
            markdown_value(row.get("peak_gpu_mb"), 1),
        ]
        lines.append("| " + " | ".join(values) + " |")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


ALGORITHM_LABELS = {
    "FlowMatchingAlgorithm": "FM",
    "FlowMatchingLognormAlgorithm": "FM-LN",
    "MeanFlowAlgorithm": "Mean Flow",
    "MeanFlowDistillAlgorithm": "MF-Distill",
    "ConsistencyAlgorithm": "Consistency",
    "ReflowAlgorithm": "Reflow",
}


def experiment_labels(records: list[dict[str, Any]], dataset: str) -> dict[str, str]:
    """Build stable labels from recorded algorithm classes and config metadata."""
    labels: dict[str, str] = {}
    for row in records:
        if row.get("dataset") != dataset:
            continue
        experiment = str(row.get("experiment", "unknown"))
        algorithm_class = str(row.get("algorithm_class", ""))
        base = ALGORITHM_LABELS.get(algorithm_class, str(row.get("experiment_name", experiment)))
        # Preserve meaningful variant names without exposing the redundant
        # dataset suffix already represented by the plot title.
        configured = str(row.get("experiment_name", ""))
        canonical = {"fm", "fm_lognorm", "mf", "mf_distill", "consistency", "reflow"}
        labels[experiment] = base if configured in canonical else f"{base} [{configured}]"
    return labels


def plot_primary_loss_curves(
    records: list[dict[str, Any]], dataset: str, out_path: Path
) -> None:
    """Plot one log-scale training-loss curve per discovered dataset run."""
    plt.figure()
    for experiment, label in experiment_labels(records, dataset).items():
        by_epoch: dict[int, dict[str, Any]] = {}
        for row in records:
            if (
                row.get("experiment") == experiment
                and row.get("record_type") == "train_epoch"
                and row.get("epoch") is not None
                and row.get("loss") is not None
            ):
                by_epoch[int(row["epoch"])] = row
        if not by_epoch:
            continue
        epochs = sorted(by_epoch)
        plt.plot(
            epochs,
            [by_epoch[epoch]["loss"] for epoch in epochs],
            label=label,
        )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.title(f"{dataset_label(dataset)} Training Loss Curves")
    if plt.gca().has_data():
        plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def build_training_cost_table(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build one training-cost row per canonical experiment directory."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(
            str(record.get("experiment", "unknown")), []
        ).append(record)

    rows: list[dict[str, Any]] = []
    for experiment, experiment_records in sorted(grouped.items()):
        training = sorted(
            (
                row for row in experiment_records
                if row.get("record_type") == "train_epoch"
                and row.get("epoch") is not None
            ),
            key=lambda row: row["epoch"],
        )
        if not training:
            continue
        evaluations = [
            row for row in experiment_records
            if row.get("record_type") == "evaluation"
        ]
        final_training = training[-1]
        sample_count = next(
            (
                row.get("num_generated_samples")
                for row in reversed(evaluations)
                if row.get("num_generated_samples") is not None
            ),
            "unknown",
        )
        eval_epoch = max(
            (row["epoch"] for row in evaluations if row.get("epoch") is not None),
            default=None,
        )
        training_seconds = final_training.get("training_time")
        rows.append({
            "experiment": experiment,
            "algorithm": next(
                (
                    row.get("algorithm_class") for row in experiment_records
                    if row.get("algorithm_class")
                ),
                None,
            ),
            "epochs": final_training.get("epoch"),
            "total_training_hours": (
                training_seconds / 3600 if training_seconds is not None else None
            ),
            "peak_gpu_mb": max(
                (row.get("peak_gpu_memory") or 0 for row in training),
                default=None,
            ),
            "num_generated_samples": sample_count,
            "eval_epoch": eval_epoch,
        })
    return rows


def plot_fid_subset(
    records: list[dict[str, Any]],
    experiments: dict[str, str],
    out_path: Path,
    title: str,
) -> None:
    """Render a controlled FID comparison without unrelated runs."""
    plt.figure()
    for experiment, label in experiments.items():
        matching = [
            row for row in records
            if row.get("experiment") == experiment
            and row.get("record_type") == "evaluation"
            and row.get("fid") is not None
        ]
        comparisons = sorted(
            {str(row.get("comparison") or experiment) for row in matching}
        )
        for comparison in comparisons:
            rows = sorted(
                (
                    row for row in matching
                    if str(row.get("comparison") or experiment) == comparison
                ),
                key=lambda row: row.get("nfe") or -1,
            )
            series_label = label
            if comparison != experiment:
                series_label = f"{label} [{comparison.removeprefix(experiment + '@')}]"
            plt.plot(
                [row["nfe"] for row in rows],
                [row["fid"] for row in rows],
                marker="o",
                label=series_label,
            )
    plt.xlabel("NFE")
    plt.ylabel("FID")
    plt.title(title)
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def dataset_label(dataset: str) -> str:
    return {"cifar10": "CIFAR-10", "celeba": "CelebA"}.get(
        dataset, dataset.replace("_", " ").title()
    )


def render_plots(combined_path: Path, plot_dir: Path, records: list[dict[str, Any]]) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    record_types = {row.get("record_type") for row in records}
    if "training" in record_types or "train_epoch" in record_types:
        plot_loss_vs_epoch(str(combined_path), str(plot_dir / "loss_vs_epoch.png"))
        plot_training_time_comparison(
            str(combined_path), str(plot_dir / "training_time.png")
        )
        plot_gpu_memory_comparison(
            str(combined_path), str(plot_dir / "gpu_memory.png")
        )
        for dataset in sorted({str(row.get("dataset")) for row in records} - {"unknown"}):
            plot_primary_loss_curves(
                records, dataset, plot_dir / f"loss_curves_{dataset}.png"
            )
    if "sampling" in record_types:
        plot_sampling_time_vs_nfe(
            str(combined_path), str(plot_dir / "sampling_time_vs_nfe.png")
        )
    if "evaluation" in record_types:
        for dataset in sorted({str(row.get("dataset")) for row in records} - {"unknown"}):
            labels = experiment_labels(records, dataset)
            if labels:
                plot_fid_subset(
                    records,
                    labels,
                    plot_dir / f"fid_vs_nfe_{dataset}.png",
                    f"{dataset_label(dataset)}: FID vs NFE",
                )
        plot_is_vs_nfe(str(combined_path), str(plot_dir / "is_vs_nfe.png"))
    if {"sampling", "evaluation"} <= record_types:
        plot_fid_vs_sampling_time(
            str(combined_path), str(plot_dir / "fid_vs_sampling_time.png")
        )


def main() -> int:
    args = parse_args()
    results_root = Path(args.results_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    markdown_path = Path(args.markdown_output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_path = output_dir / "combined_metrics.jsonl"

    # Select only the canonical JSONL for each run directory: the file whose
    # stem matches the run directory name.  This prevents auxiliary files
    # (e.g. smoke_lognorm.jsonl inside fm_lognorm_rtx3060/metrics/) from
    # overwriting the real run's evaluation records during deduplication.
    all_jsonl = sorted(
        path for path in results_root.glob("*/metrics/*.jsonl")
        if path.resolve() != combined_path
        and output_dir not in path.resolve().parents
    )
    inputs: list[Path] = []
    seen_run_dirs: set[Path] = set()
    for path in all_jsonl:
        run_dir = path.parent.parent
        if path.stem == run_dir.name:
            inputs.append(path)
            seen_run_dirs.add(run_dir)
    # Warn for any run directory that has metrics files but no canonical one
    for path in all_jsonl:
        run_dir = path.parent.parent
        if run_dir not in seen_run_dirs:
            print(
                f"  [WARN] {run_dir.name}: no canonical JSONL found "
                f"(stem must equal run dir name); skipped {path.name}",
                file=sys.stderr,
            )
            seen_run_dirs.add(run_dir)  # only warn once per dir
    if not inputs:
        print(f"No metric JSONL files found under {results_root}", file=sys.stderr)
        return 1

    records = deduplicate(load_records(inputs))
    if not records:
        print("Metric files contained no records.", file=sys.stderr)
        return 1

    with combined_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    summary = build_summary(records)
    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    training_cost_rows = build_training_cost_table(records)
    training_cost_path = output_dir / "training_cost_table.csv"
    if training_cost_rows:
        with training_cost_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(training_cost_rows[0])
            )
            writer.writeheader()
            writer.writerows(training_cost_rows)

    write_markdown_summary(markdown_path, summary, training_cost_rows)

    render_plots(combined_path, output_dir / "plots", records)
    print(f"Aggregated {len(records)} records from {len(inputs)} files.")
    print(f"Combined metrics: {combined_path}")
    print(f"Summary table:    {summary_path}")
    print(f"Training costs:   {training_cost_path}")
    print(f"Markdown summary: {markdown_path}")
    print(f"Plots:            {output_dir / 'plots'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
