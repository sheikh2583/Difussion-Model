"""Numbered, non-destructive checkpoint-run layout helpers.

Layout::

    <experiment>/checkpoints/run_1/*.pt
    <experiment>/checkpoints/run_1/archive/*.zip
    <experiment>/checkpoints/run_2/*.pt
    <experiment>/checkpoints/run_2/archive/*.zip

The underscore form is intentional: it is safe to quote and sort on Windows,
Linux, PowerShell, Bash, and Python. Legacy flat layouts remain readable and
can be migrated without overwriting any file.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Iterable, Optional


RUN_DIRECTORY_PATTERN = re.compile(r"^run_(\d+)$")
CHECKPOINT_EPOCH_PATTERN = re.compile(r"_epoch(\d+)\.pt$")


def checkpoint_root(run_dir: Path) -> Path:
    return Path(run_dir) / "checkpoints"


def checkpoint_run_directory(run_dir: Path, run_number: int) -> Path:
    if run_number < 1:
        raise ValueError("checkpoint run number must be at least 1")
    return checkpoint_root(Path(run_dir)) / f"run_{run_number}"


def checkpoint_run_number_from_path(path: Path) -> Optional[int]:
    match = RUN_DIRECTORY_PATTERN.fullmatch(Path(path).parent.name)
    return int(match.group(1)) if match else None


def numbered_run_directories(run_dir: Path) -> list[tuple[int, Path]]:
    root = checkpoint_root(Path(run_dir))
    found: list[tuple[int, Path]] = []
    if root.is_dir():
        for path in root.iterdir():
            match = RUN_DIRECTORY_PATTERN.fullmatch(path.name)
            if path.is_dir() and match:
                found.append((int(match.group(1)), path))
    return sorted(found)


def latest_checkpoint_run_number(run_dir: Path) -> Optional[int]:
    numbered = numbered_run_directories(run_dir)
    if numbered:
        return numbered[-1][0]
    root = checkpoint_root(Path(run_dir))
    if root.is_dir() and any(root.glob("*.pt")):
        return 1
    return None


def latest_checkpoint_run_directory(run_dir: Path) -> Optional[Path]:
    numbered = numbered_run_directories(run_dir)
    if numbered:
        return numbered[-1][1]
    root = checkpoint_root(Path(run_dir))
    return root if root.is_dir() and any(root.glob("*.pt")) else None


def _history_run_directories(run_dir: Path) -> Iterable[Path]:
    history = Path(run_dir).parent / "history"
    if history.is_dir():
        yield from (
            path for path in history.glob(f"{Path(run_dir).name}_*") if path.is_dir()
        )


def next_checkpoint_run_number(run_dir: Path) -> int:
    numbers: list[int] = []
    current = latest_checkpoint_run_number(run_dir)
    if current is not None:
        numbers.append(current)
    for historical in _history_run_directories(Path(run_dir)):
        number = latest_checkpoint_run_number(historical)
        if number is not None:
            numbers.append(number)
    return max(numbers, default=0) + 1


def find_checkpoint(
    run_dir: Path,
    class_name: str,
    epoch: int,
    run_number: Optional[int] = None,
) -> Optional[Path]:
    filename = f"{class_name}_epoch{epoch}.pt"
    if run_number is not None:
        candidate = checkpoint_run_directory(Path(run_dir), run_number) / filename
        return candidate if candidate.is_file() else None
    directory = latest_checkpoint_run_directory(Path(run_dir))
    if directory is None:
        return None
    candidate = directory / filename
    return candidate if candidate.is_file() else None


def latest_epoch_checkpoint(
    run_dir: Path,
    class_name: Optional[str] = None,
    maximum_epoch: Optional[int] = None,
    run_number: Optional[int] = None,
) -> Optional[tuple[int, Path]]:
    directory = (
        checkpoint_run_directory(Path(run_dir), run_number)
        if run_number is not None
        else latest_checkpoint_run_directory(Path(run_dir))
    )
    if directory is None or not directory.is_dir():
        return None
    prefix = f"{class_name}_epoch" if class_name else ""
    candidates: list[tuple[int, Path]] = []
    for path in directory.glob(f"{prefix}*.pt"):
        match = CHECKPOINT_EPOCH_PATTERN.search(path.name)
        if match:
            epoch = int(match.group(1))
            if maximum_epoch is None or epoch <= maximum_epoch:
                candidates.append((epoch, path))
    return max(candidates, key=lambda item: item[0]) if candidates else None


def legacy_migration_plan(run_dir: Path) -> list[tuple[Path, Path]]:
    root = checkpoint_root(Path(run_dir))
    if not root.is_dir():
        return []
    target = checkpoint_run_directory(Path(run_dir), 1)
    plan = [(path, target / path.name) for path in sorted(root.glob("*.pt"))]
    legacy_archive = root / "archive"
    if legacy_archive.is_dir():
        plan.extend(
            (path, target / "archive" / path.name)
            for path in sorted(legacy_archive.glob("*.zip"))
        )
    return plan


def migrate_legacy_checkpoint_layout(run_dir: Path) -> list[tuple[Path, Path]]:
    """Move flat legacy files into run_1, refusing every name collision."""
    plan = legacy_migration_plan(Path(run_dir))
    collisions = [destination for _, destination in plan if destination.exists()]
    if collisions:
        listed = "\n- ".join(str(path) for path in collisions)
        raise FileExistsError(
            "Checkpoint migration refused to overwrite existing files:\n- " + listed
        )
    for source, destination in plan:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
    legacy_archive = checkpoint_root(Path(run_dir)) / "archive"
    if legacy_archive.is_dir() and not any(legacy_archive.iterdir()):
        legacy_archive.rmdir()
    return plan


def run_directory_from_checkpoint(checkpoint: Path) -> Optional[Path]:
    """Infer an experiment directory from legacy or numbered checkpoint paths."""
    checkpoint = Path(checkpoint).resolve()
    parent = checkpoint.parent
    if RUN_DIRECTORY_PATTERN.fullmatch(parent.name) and parent.parent.name == "checkpoints":
        return parent.parent.parent
    if parent.name == "checkpoints":
        return parent.parent
    return None
