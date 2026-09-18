"""Safe run-directory lifecycle helpers shared by training front ends."""

from __future__ import annotations

import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Protocol, Tuple


class _DatasetConfig(Protocol):
    name: str


class _ExperimentConfig(Protocol):
    experiment_name: str
    output_dir: str
    dataset: _DatasetConfig


_CHECKPOINT_EPOCH = re.compile(r"_epoch(\d+)\.pt$")


def _validate_component(value: str, label: str) -> str:
    """Reject path-like config values before using them as a run name."""
    if not value or value in {".", ".."}:
        raise ValueError(f"{label} must be a non-empty name")
    if Path(value).name != value or "/" in value or "\\" in value:
        raise ValueError(f"{label} must not contain path separators: {value!r}")
    return value


def run_directory(cfg: _ExperimentConfig, project_root: Path) -> Path:
    """Return the canonical ``<output>/<experiment>_<dataset>`` directory."""
    experiment = _validate_component(cfg.experiment_name, "experiment_name")
    dataset = _validate_component(cfg.dataset.name, "dataset.name")
    output = Path(cfg.output_dir)
    if not output.is_absolute():
        output = project_root / output
    return output.resolve() / f"{experiment}_{dataset}"


def has_run_artifacts(run_dir: Path) -> bool:
    """Whether a canonical run directory contains anything worth preserving."""
    return run_dir.is_dir() and any(run_dir.iterdir())


def latest_checkpoint(run_dir: Path) -> Optional[Tuple[int, Path]]:
    """Find the numerically latest epoch checkpoint in a run directory."""
    checkpoint_dir = run_dir / "checkpoints"
    candidates: list[Tuple[int, Path]] = []
    if checkpoint_dir.is_dir():
        for path in checkpoint_dir.glob("*.pt"):
            match = _CHECKPOINT_EPOCH.search(path.name)
            if match:
                candidates.append((int(match.group(1)), path))
    return max(candidates, key=lambda item: item[0]) if candidates else None


def archive_existing_run(run_dir: Path, timestamp: Optional[str] = None) -> Optional[Path]:
    """Move an existing run into timestamped history and return its new path.

    This is intentionally recoverable: fresh training never deletes or overwrites
    the previous canonical run directory.
    """
    if not has_run_artifacts(run_dir):
        return None

    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    history_dir = run_dir.parent / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    destination = history_dir / f"{run_dir.name}_{timestamp}"
    counter = 1
    while destination.exists():
        destination = history_dir / f"{run_dir.name}_{timestamp}_{counter}"
        counter += 1

    shutil.move(str(run_dir), str(destination))
    return destination
