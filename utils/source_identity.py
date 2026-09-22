"""Content identity for training-relevant source inputs."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SOURCE_DIRS = (
    "algorithms", "models", "training", "data", "evaluation", "sampling", "codec"
)
SOURCE_FILES = (
    "train.py",
    "config/config.py",
    "experiments/runner.py",
    "utils/checkpoint_provenance.py",
    "utils/checkpoint_runs.py",
    "utils/checkpoints.py",
    "utils/device.py",
    "utils/gpu_lock.py",
    "utils/run_environment.py",
    "utils/run_lifecycle.py",
    "utils/source_identity.py",
    "scripts/checkpoint_path.py",
    "scripts/generate_reflow_pairs.py",
    "scripts/generate_reflow_pairs_latent.py",
    "scripts/workflow_guard.py",
    "scripts/linux/workflow_guard.sh",
)


class SourceIdentityError(RuntimeError):
    """Raised when frozen training inputs no longer match the live tree."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def training_input_paths(
    project_root: Path,
    *,
    configs: Iterable[Path | str] = (),
    launchers: Iterable[Path | str] = (),
) -> list[Path]:
    """Return the stable, documentation-free source set for planned jobs."""
    root = project_root.resolve()
    paths: set[Path] = set()
    for directory in SOURCE_DIRS:
        base = root / directory
        if base.is_dir():
            paths.update(path for path in base.rglob("*.py") if path.is_file())
    paths.update(root / name for name in SOURCE_FILES if (root / name).is_file())
    for value in (*configs, *launchers):
        path = Path(value)
        path = path if path.is_absolute() else root / path
        if not path.is_file():
            raise FileNotFoundError(f"Training identity input does not exist: {path}")
        paths.add(path.resolve())
    return sorted(paths)


def build_source_manifest(
    project_root: Path,
    *,
    configs: Iterable[Path | str] = (),
    launchers: Iterable[Path | str] = (),
) -> dict[str, Any]:
    root = project_root.resolve()
    config_inputs = _relative_inputs(root, configs)
    launcher_inputs = _relative_inputs(root, launchers)
    files = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in training_input_paths(
            root, configs=config_inputs, launchers=launcher_inputs
        )
    }
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":"))
    return {
        "schema_version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_identity_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "inputs": {
            "configs": config_inputs,
            "launchers": launcher_inputs,
        },
        "files": files,
    }


def _relative_inputs(root: Path, values: Iterable[Path | str]) -> list[str]:
    """Normalize explicit inputs so verification can reconstruct the file set."""
    relative: set[str] = set()
    for value in values:
        path = Path(value)
        resolved = (path if path.is_absolute() else root / path).resolve()
        try:
            relative.add(resolved.relative_to(root).as_posix())
        except ValueError as exc:
            raise ValueError(
                f"Training identity input is outside the project root: {resolved}"
            ) from exc
    return sorted(relative)


def write_source_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.",
        suffix=".tmp", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def verify_source_manifest(project_root: Path, manifest: dict[str, Any]) -> None:
    root = project_root.resolve()
    expected = manifest.get("files")
    if not isinstance(expected, dict):
        raise SourceIdentityError("Source manifest has no valid files mapping")
    inputs = manifest.get("inputs")
    if isinstance(inputs, dict):
        configs = inputs.get("configs", [])
        launchers = inputs.get("launchers", [])
        if not isinstance(configs, list) or not isinstance(launchers, list):
            raise SourceIdentityError("Source manifest has invalid explicit inputs")
        # Missing explicit inputs are reported below as deletions instead of
        # escaping as a generic FileNotFoundError.
        existing_configs = [value for value in configs if (root / value).is_file()]
        existing_launchers = [value for value in launchers if (root / value).is_file()]
        current_paths = training_input_paths(
            root, configs=existing_configs, launchers=existing_launchers
        )
        current = {
            path.relative_to(root).as_posix(): sha256_file(path)
            for path in current_paths
        }
    else:
        # Schema-v1 manifests did not preserve enough scope information to
        # detect additions.  Retain content/deletion validation for them.
        current = {
            relative: sha256_file(root / relative)
            for relative in expected
            if (root / relative).is_file()
        }

    changed: list[str] = []
    for relative in sorted(set(expected) | set(current)):
        if relative not in current:
            changed.append(f"{relative} (deleted)")
        elif relative not in expected:
            changed.append(f"{relative} (added)")
        elif current[relative] != expected[relative]:
            changed.append(f"{relative} (content changed)")
    if changed:
        raise SourceIdentityError(
            "Training-relevant inputs changed after suite startup; refusing the next "
            "GPU job:\n  - " + "\n  - ".join(changed)
        )
