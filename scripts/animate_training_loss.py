"""Animate training-loss curves stored in experiment JSONL files."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.animation as animation
import matplotlib.pyplot as plt


# Default: first JSONL file found under results/ (auto-discovered at runtime)
# Override by passing one or more paths as positional arguments.
DEFAULT_INPUT: Path | None = None
DEFAULT_OUTPUT = Path("results/training_loss_animated.gif")
COLORS = ("#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FFC107")


def readable_name(name: str) -> str:
    """Convert names such as FlowMatchingAlgorithm to Flow Matching."""
    name = name.removesuffix("Algorithm")
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name).strip()


def load_train_records(jsonl_path: Path) -> tuple[str, list[dict]]:
    """Load training records, keeping the latest record for each epoch."""
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"Metrics file not found: {jsonl_path}")

    records_by_epoch: dict[int, dict] = {}
    algorithm_name = jsonl_path.stem

    with jsonl_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {jsonl_path} at line {line_number}: {error}"
                ) from error

            if record.get("record_type") != "train_epoch":
                continue
            if record.get("epoch") is None or record.get("loss") is None:
                continue

            algorithm_name = record.get("algorithm", algorithm_name)
            records_by_epoch[int(record["epoch"])] = record

    records = sorted(records_by_epoch.values(), key=lambda row: row["epoch"])
    if not records:
        raise ValueError(f"No usable train_epoch records found in {jsonl_path}")

    return readable_name(algorithm_name), records


def create_animation(
    input_paths: list[Path],
    output_path: Path,
    fps: int = 10,
    interval_ms: int = 120,
) -> None:
    series = [load_train_records(path) for path in input_paths]
    all_records = [record for _, records in series for record in records]

    epochs = [int(record["epoch"]) for record in all_records]
    losses = [float(record["loss"]) for record in all_records]
    min_epoch, max_epoch = min(epochs), max(epochs)
    min_loss, max_loss = min(losses), max(losses)
    loss_padding = max((max_loss - min_loss) * 0.1, max_loss * 0.05, 1e-6)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(min_epoch, max(max_epoch, min_epoch + 1))
    ax.set_ylim(max(0.0, min_loss - loss_padding), max_loss + loss_padding)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title("Training Loss Progress")
    ax.grid(True, alpha=0.3)

    lines = []
    points = []
    for index, (label, _) in enumerate(series):
        color = COLORS[index % len(COLORS)]
        (line,) = ax.plot([], [], label=label, color=color, linewidth=2.2)
        (point,) = ax.plot([], [], "o", color=color, markersize=6)
        lines.append(line)
        points.append(point)
    ax.legend()

    epoch_text = ax.text(
        0.98,
        0.95,
        "",
        transform=ax.transAxes,
        horizontalalignment="right",
        verticalalignment="top",
        fontsize=11,
    )

    def update(frame: int):
        point_count = frame + 1
        latest_epochs = []
        for (_, records), line, point in zip(series, lines, points):
            visible = records[:point_count]
            x_values = [record["epoch"] for record in visible]
            y_values = [record["loss"] for record in visible]
            line.set_data(x_values, y_values)
            point.set_data(x_values[-1:], y_values[-1:])
            latest_epochs.append(x_values[-1])
        epoch_text.set_text(f"Epoch {max(latest_epochs)}")
        return *lines, *points, epoch_text

    frame_count = max(len(records) for _, records in series)
    loss_animation = animation.FuncAnimation(
        fig,
        update,
        frames=frame_count,
        interval=interval_ms,
        blit=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    loss_animation.save(output_path, writer="pillow", fps=fps, dpi=120)
    plt.close(fig)
    print(f"Saved: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an animated loss curve from one or more metrics files."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        default=None,
        help=(
            "JSONL metric files to animate. "
            "If omitted, auto-discovers the first *.jsonl under results/."
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--interval-ms", type=int, default=120)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    inputs = args.inputs
    if not inputs:
        # Auto-discover: find all *.jsonl under results/
        results_root = Path("results")
        inputs = sorted(results_root.rglob("*.jsonl")) if results_root.exists() else []
        if not inputs:
            print("No JSONL metric files found under results/. Train a model first.")
            raise SystemExit(1)
        print(f"Auto-discovered {len(inputs)} JSONL file(s):")
        for p in inputs:
            print(f"  {p}")
    create_animation(inputs, args.output, args.fps, args.interval_ms)
