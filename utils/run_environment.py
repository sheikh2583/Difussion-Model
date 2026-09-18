"""Capture reproducibility metadata for cross-machine experiment comparison."""

from __future__ import annotations

import json
import hashlib
import os
import platform
import re
import socket
import subprocess
import sys
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import torch


def _machine_label_token(value: str) -> str:
    """Normalize hardware/host text into a readable label component."""
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", value.lower())).strip("-")


def _automatic_machine_label(
    os_name: str,
    hostname: str,
    gpu_name: Optional[str],
    gpu_memory_gb: Optional[int],
) -> str:
    """Build a stable, descriptive label without requiring user setup."""
    parts = [_machine_label_token(os_name), _machine_label_token(hostname)]
    if gpu_name:
        parts.append(_machine_label_token(gpu_name))
        if gpu_memory_gb is not None:
            parts.append(f"{gpu_memory_gb}gb")
    else:
        parts.append("cpu")
    return "-".join(part for part in parts if part)


def add_config_hash(metadata: dict[str, Any], config: Any) -> dict[str, Any]:
    """Return a copy of ``metadata`` carrying the canonical config digest."""
    config_value = asdict(config) if is_dataclass(config) else config
    serialized = json.dumps(
        config_value, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    enriched = dict(metadata)
    enriched["config_sha256"] = hashlib.sha256(serialized).hexdigest()
    return enriched


def _git_value(project_root: Path, *args: str) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def collect_run_environment(
    project_root: Path,
    machine_label: Optional[str] = None,
) -> dict[str, Any]:
    """Return flat, JSON-safe metadata suitable for every result record."""
    hostname = socket.gethostname()
    os_name = platform.system()
    commit = _git_value(project_root, "rev-parse", "HEAD")
    dirty_output = _git_value(
        project_root, "status", "--porcelain", "--untracked-files=no"
    )
    tracked_diff = _git_value(project_root, "diff", "--binary", "HEAD")
    diff_sha256 = (
        hashlib.sha256(tracked_diff.encode("utf-8")).hexdigest()
        if tracked_diff else None
    )
    code_identity = commit
    if commit and diff_sha256:
        code_identity = f"{commit}+dirty:{diff_sha256[:12]}"
    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
    gpu_memory_gb = (
        round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3))
        if cuda_available
        else None
    )
    label = (
        machine_label
        or os.environ.get("DIFFUSION_MACHINE_LABEL")
        or _automatic_machine_label(os_name, hostname, gpu_name, gpu_memory_gb)
    )
    return {
        "session_id": uuid.uuid4().hex,
        "session_started_utc": datetime.now(timezone.utc).isoformat(),
        "machine_label": label,
        "hostname": hostname,
        "os_name": os_name,
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": gpu_name,
        "gpu_memory_gb": gpu_memory_gb,
        "gpu_count": torch.cuda.device_count() if cuda_available else 0,
        "git_commit": commit,
        "git_dirty": bool(dirty_output) if dirty_output is not None else None,
        "git_diff_sha256": diff_sha256,
        "code_identity": code_identity,
        "command": sys.argv,
    }


def write_run_environment(run_dir: Path, metadata: dict[str, Any]) -> None:
    """Publish latest metadata and append it to the run's session history."""
    run_dir.mkdir(parents=True, exist_ok=True)
    latest = run_dir / "run_environment.json"
    temporary = latest.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    os.replace(temporary, latest)
    with (run_dir / "run_environment.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metadata, sort_keys=True) + "\n")
