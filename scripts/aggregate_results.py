#!/usr/bin/env python3
"""Merge experiment JSONL metrics, build the thesis table, and render plots."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from utils.plots import (
    plot_fid_vs_nfe,
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
    return parser.parse_args()


def load_records(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
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
                experiment = path.parent.parent.name
                record["algorithm_class"] = record.get("algorithm")
                record["experiment"] = experiment
                # Plotting utilities group on `algorithm`; use the run name so
                # CIFAR-10 and CelebA measurements cannot be merged silently.
                record["algorithm"] = experiment
                records.append(record)
    return records


def deduplicate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the latest row for each logical experiment measurement."""
    latest: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in records:
        key = (
            record.get("experiment"),
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
        last_training = training[-1] if training else {}
        rows.append({
            "experiment": algorithm,
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
            "fid_at_1": (evaluations.get(1) or {}).get("fid"),
            "fid_at_5": (evaluations.get(5) or {}).get("fid"),
            "fid_at_20": (evaluations.get(20) or {}).get("fid"),
            "is_at_1": (evaluations.get(1) or {}).get("is_mean"),
            "is_at_5": (evaluations.get(5) or {}).get("is_mean"),
            "is_at_20": (evaluations.get(20) or {}).get("is_mean"),
        })
    return rows


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
    if "sampling" in record_types:
        plot_sampling_time_vs_nfe(
            str(combined_path), str(plot_dir / "sampling_time_vs_nfe.png")
        )
    if "evaluation" in record_types:
        plot_fid_vs_nfe(str(combined_path), str(plot_dir / "fid_vs_nfe.png"))
        plot_is_vs_nfe(str(combined_path), str(plot_dir / "is_vs_nfe.png"))
    if {"sampling", "evaluation"} <= record_types:
        plot_fid_vs_sampling_time(
            str(combined_path), str(plot_dir / "fid_vs_sampling_time.png")
        )


def main() -> int:
    args = parse_args()
    results_root = Path(args.results_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_path = output_dir / "combined_metrics.jsonl"

    inputs = sorted(
        path for path in results_root.glob("*/metrics/*.jsonl")
        if path.resolve() != combined_path
        and output_dir not in path.resolve().parents
    )
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

    render_plots(combined_path, output_dir / "plots", records)
    print(f"Aggregated {len(records)} records from {len(inputs)} files.")
    print(f"Combined metrics: {combined_path}")
    print(f"Summary table:    {summary_path}")
    print(f"Plots:            {output_dir / 'plots'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
