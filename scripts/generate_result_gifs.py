#!/usr/bin/env python3
"""Generate thesis animations from existing metrics, configs, and filenames.

This script is deliberately read-only with respect to training artifacts. It
does not import model code, open checkpoint tensors, sample, or evaluate. A
checkpoint contributes only the epoch parsed from its ``*_epoch<N>.pt`` name.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.animation as animation
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METRICS = PROJECT_ROOT / "results" / "aggregate" / "combined_metrics.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "aggregate" / "animations"
CHECKPOINT_EPOCH = re.compile(r"epoch(\d+)")


def experiment_dataset(results_root: Path, experiment: str) -> str | None:
    config_path = results_root / experiment / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        config = {}
    dataset = config.get("dataset", {})
    if isinstance(dataset, dict) and dataset.get("name"):
        return str(dataset["name"]).lower()
    return next(
        (name for name in ("cifar10", "celeba") if experiment.endswith(f"_{name}")),
        None,
    )


def dataset_experiments(
    results_root: Path, records: list[dict[str, Any]], dataset: str
) -> tuple[str, ...]:
    experiments = {str(row["experiment"]) for row in records if row.get("experiment")}
    return tuple(
        sorted(
            experiment
            for experiment in experiments
            if experiment_dataset(results_root, experiment) == dataset
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create FID/NFE, 3D FID/NFE/epoch, and epoch/loss GIFs from "
            "existing aggregate data. No checkpoints are loaded."
        )
    )
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument(
        "--results-root", type=Path, default=PROJECT_ROOT / "results"
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--dataset",
        choices=("cifar10", "celeba", "celeba_latent", "all"),
        default="all",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=None,
        help="Override the default experiments for the selected dataset.",
    )
    parser.add_argument(
        "--max-nfe",
        type=int,
        default=20,
        help="Maximum NFE shown in FID comparisons (default: 20).",
    )
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=90)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Aggregate metrics not found: {path}\n"
            "Run: venv/bin/python scripts/aggregate_results.py"
        )
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
            records.append(row)
    return records


def config_label(results_root: Path, experiment: str, records: list[dict]) -> str:
    config_path = results_root / experiment / "config.json"
    config: dict[str, Any] = {}
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            config = {}

    algorithm_class = next(
        (
            row.get("algorithm_class")
            for row in records
            if row.get("experiment") == experiment and row.get("algorithm_class")
        ),
        experiment,
    )
    short_names = {
        "FlowMatchingAlgorithm": "FM",
        "FlowMatchingLognormAlgorithm": "FM + Logit-Normal",
        "MeanFlowAlgorithm": "Mean Flow",
        "MeanFlowDistillAlgorithm": "MF-Distill",
        "ConsistencyAlgorithm": "Consistency",
        "ReflowAlgorithm": "Reflow",
    }
    algorithm = short_names.get(str(algorithm_class))
    if algorithm is None:
        algorithm = str(algorithm_class).removesuffix("Algorithm")
        algorithm = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", algorithm)
    return algorithm


def checkpoint_epochs(results_root: Path, experiments: set[str]) -> dict[str, list[int]]:
    epochs: dict[str, list[int]] = {}
    for experiment in sorted(experiments):
        found: set[int] = set()
        checkpoint_root = results_root / experiment / "checkpoints"
        if checkpoint_root.is_dir():
            for path in checkpoint_root.glob("run_*/*.pt"):
                match = CHECKPOINT_EPOCH.search(path.name)
                if match:
                    found.add(int(match.group(1)))
        epochs[experiment] = sorted(found)
    return epochs


def metric_series(
    records: list[dict[str, Any]], experiments: set[str], max_nfe: int
) -> tuple[dict[str, dict[tuple[int, int], float]], dict[str, dict[int, float]]]:
    fid: dict[str, dict[tuple[int, int], float]] = defaultdict(dict)
    loss: dict[str, dict[int, float]] = defaultdict(dict)
    for row in records:
        experiment = row.get("experiment")
        if experiment not in experiments or row.get("epoch") is None:
            continue
        epoch = int(row["epoch"])
        if (
            row.get("record_type") == "evaluation"
            and row.get("nfe") is not None
            and int(row["nfe"]) <= max_nfe
            and row.get("fid") is not None
        ):
            fid[experiment][(epoch, int(row["nfe"]))] = float(row["fid"])
        elif row.get("record_type") == "train_epoch" and row.get("loss") is not None:
            loss[experiment][epoch] = float(row["loss"])
    return dict(fid), dict(loss)


def save_fid_vs_nfe(
    fid: dict[str, dict[tuple[int, int], float]],
    labels: dict[str, str],
    output: Path,
    fps: int,
    dpi: int,
    dataset_label: str,
) -> None:
    max_epoch = max(epoch for series in fid.values() for epoch, _ in series)
    all_nfes = [nfe for series in fid.values() for _, nfe in series]
    all_fids = [value for series in fid.values() for value in series.values()]
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.colormaps["tab10"]

    def update(epoch: int):
        ax.clear()
        for index, (experiment, series) in enumerate(sorted(fid.items())):
            available_epochs = [seen for seen, _ in series if seen <= epoch]
            if not available_epochs:
                continue
            latest = max(available_epochs)
            points = sorted(
                (nfe, value)
                for (seen, nfe), value in series.items()
                if seen == latest
            )
            ax.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                marker="o",
                linewidth=2.5,
                color=colors(index % 10),
                label=labels[experiment],
            )
            annotated_points = points if len(fid) <= 3 else points[-1:]
            for point_index, (nfe, value) in enumerate(annotated_points):
                ax.annotate(
                    f"{value:.1f}",
                    (nfe, value),
                    xytext=(5, 7 if (index + point_index) % 2 == 0 else -13),
                    textcoords="offset points",
                    fontsize=7,
                    color=colors(index % 10),
                )
        ax.set_xlim(min(all_nfes) - 0.8, max(all_nfes) + 1.2)
        ax.set_ylim(max(0, min(all_fids) * 0.82), max(all_fids) * 1.08)
        ax.set_xticks(sorted(set(all_nfes)))
        ax.set_xlabel("Number of Function Evaluations (NFE)")
        ax.set_ylabel("FID (lower is better)")
        ax.set_title(
            f"{dataset_label}: Sample Quality vs Inference Cost — through epoch {epoch}"
        )
        ax.grid(alpha=0.35, linestyle="--")
        handles, legend_labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, legend_labels, fontsize=8, loc="best", framealpha=0.9)

    movie = animation.FuncAnimation(fig, update, frames=range(1, max_epoch + 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    movie.save(output, writer="pillow", fps=fps, dpi=dpi)
    plt.close(fig)


def save_fid_nfe_epoch_3d(
    fid: dict[str, dict[tuple[int, int], float]],
    labels: dict[str, str],
    output: Path,
    fps: int,
    dpi: int,
    dataset_label: str,
) -> None:
    max_epoch = max(epoch for series in fid.values() for epoch, _ in series)
    all_fids = [value for series in fid.values() for value in series.values()]
    fig = plt.figure(figsize=(9, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    colors = plt.colormaps["tab10"]

    def update(epoch: int):
        ax.clear()
        for index, (experiment, series) in enumerate(sorted(fid.items())):
            points = sorted(
                (seen, nfe, value)
                for (seen, nfe), value in series.items()
                if seen <= epoch
            )
            if not points:
                continue
            ax.scatter(
                [point[1] for point in points],
                [point[0] for point in points],
                [point[2] for point in points],
                s=20,
                color=colors(index % 10),
                label=labels[experiment],
            )
            by_epoch: dict[int, list[tuple[int, float]]] = defaultdict(list)
            for seen, nfe, value in points:
                by_epoch[seen].append((nfe, value))
            for seen, epoch_points in by_epoch.items():
                epoch_points.sort()
                ax.plot(
                    [point[0] for point in epoch_points],
                    [seen] * len(epoch_points),
                    [point[1] for point in epoch_points],
                    color=colors(index % 10),
                    alpha=0.65,
                )
        ax.set_xlim(left=0)
        ax.set_ylim(1, max_epoch)
        ax.set_zlim(0, max(all_fids) * 1.08)
        ax.set_xlabel("NFE")
        ax.set_ylabel("Epoch")
        ax.set_zlabel("FID")
        ax.set_title(
            f"{dataset_label}: FID by NFE and Epoch — through epoch {epoch}"
        )
        ax.view_init(elev=25, azim=35 + epoch * 1.8)
        handles, legend_labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, legend_labels, fontsize=6, loc="upper left")

    movie = animation.FuncAnimation(fig, update, frames=range(1, max_epoch + 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    movie.save(output, writer="pillow", fps=fps, dpi=dpi)
    plt.close(fig)


def save_epoch_vs_loss(
    loss: dict[str, dict[int, float]],
    checkpoints: dict[str, list[int]],
    labels: dict[str, str],
    output: Path,
    fps: int,
    dpi: int,
    dataset_label: str,
) -> None:
    max_epoch = max(epoch for series in loss.values() for epoch in series)
    positive_losses = [
        value for series in loss.values() for value in series.values() if value > 0
    ]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = plt.colormaps["tab10"]

    def update(epoch: int):
        ax.clear()
        for index, (experiment, series) in enumerate(sorted(loss.items())):
            points = sorted(
                (seen, value) for seen, value in series.items() if seen <= epoch
            )
            if not points:
                continue
            color = colors(index % 10)
            ax.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                linewidth=2,
                color=color,
                label=labels[experiment],
            )
            marked = [seen for seen in checkpoints.get(experiment, []) if seen <= epoch]
            marked_points = [(seen, series[seen]) for seen in marked if seen in series]
            if marked_points:
                ax.scatter(
                    [point[0] for point in marked_points],
                    [point[1] for point in marked_points],
                    s=22,
                    marker="D",
                    color=color,
                )
        ax.set_xlim(1, max_epoch)
        ax.set_ylim(min(positive_losses) * 0.8, max(positive_losses) * 1.25)
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Training Loss (log scale)")
        ax.set_title(
            f"{dataset_label}: Training Loss through Epoch {epoch} "
            "(diamonds = checkpoints)"
        )
        ax.grid(alpha=0.25, which="both")
        handles, legend_labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, legend_labels, fontsize=7, loc="upper right")

    movie = animation.FuncAnimation(fig, update, frames=range(1, max_epoch + 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    movie.save(output, writer="pillow", fps=fps, dpi=dpi)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if args.fps < 1 or args.dpi < 1:
        raise ValueError("--fps and --dpi must be positive integers")

    metrics_path = args.metrics.expanduser().resolve()
    results_root = args.results_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    records = load_jsonl(metrics_path)
    datasets = (
        ("cifar10", "celeba", "celeba_latent")
        if args.dataset == "all"
        else (args.dataset,)
    )
    for dataset in datasets:
        experiments = set(
            args.experiments
            or dataset_experiments(results_root, records, dataset)
        )
        if not experiments:
            print(f"[SKIP] {dataset}: no matching experiment configs")
            continue
        fid, loss = metric_series(records, experiments, args.max_nfe)
        if not fid:
            print(f"[SKIP] {dataset}: no epoch-aware FID records")
            continue
        if not loss:
            print(f"[SKIP] {dataset}: no training-loss records")
            continue

        available = set(fid) | set(loss)
        labels = {
            experiment: config_label(results_root, experiment, records)
            for experiment in available
        }
        checkpoints = checkpoint_epochs(results_root, available)
        dataset_output = output_dir / dataset
        outputs = (
            dataset_output / "nfe_vs_fid.gif",
            dataset_output / "nfe_vs_fid_vs_epoch_3d.gif",
            dataset_output / "epoch_vs_loss.gif",
        )
        dataset_label = {
            "cifar10": "CIFAR-10",
            "celeba": "CelebA (pixel)",
            "celeba_latent": "CelebA (latent)",
        }[dataset]
        save_fid_vs_nfe(
            fid, labels, outputs[0], args.fps, args.dpi, dataset_label
        )
        print(f"Saved: {outputs[0]}")
        save_fid_nfe_epoch_3d(
            fid, labels, outputs[1], args.fps, args.dpi, dataset_label
        )
        print(f"Saved: {outputs[1]}")
        save_epoch_vs_loss(
            loss, checkpoints, labels, outputs[2], args.fps, args.dpi, dataset_label
        )
        print(f"Saved: {outputs[2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
