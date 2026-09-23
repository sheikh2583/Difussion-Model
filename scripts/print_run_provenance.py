#!/usr/bin/env python3
"""Print uniform ``[run]`` provenance headers for platform launchers."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.run_environment import collect_run_environment


FIELDS = (
    "launcher_session_id",
    "hostname",
    "os_name",
    "os_release",
    "architecture",
    "python_version",
    "torch_version",
    "cuda_version",
    "cuda_visible_devices",
    "gpu_device_index",
    "gpu_uuid",
    "gpu_name",
    "gpu_memory_gb",
    "gpu_count",
    "machine_label",
    "git_commit",
    "git_dirty",
    "git_diff_sha256",
    "code_identity",
    "source_identity_sha256",
    "source_manifest",
    "parent_suite_timestamp",
    "lifecycle_mode",
    "checkpoint_series",
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text(value: object) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, bool):
        return str(value).lower()
    return str(value).replace("\r", " ").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--machine-label", required=True)
    parser.add_argument("--log-device-token", required=True)
    parser.add_argument("--log-path", required=True)
    args = parser.parse_args()

    config = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    if not config.is_file():
        raise SystemExit(f"Selected config does not exist: {config}")

    metadata = collect_run_environment(PROJECT_ROOT, args.machine_label)
    metadata["launcher_session_id"] = metadata.pop("session_id")
    metadata.update({
        "selected_config_sha256": _file_sha256(config),
        "selected_config_path": config.resolve().relative_to(PROJECT_ROOT).as_posix(),
        "log_device_token": args.log_device_token,
        "log_path": args.log_path.replace("\\", "/"),
        "git_log_policy": "eligible-not-auto-staged",
    })
    for key in (*FIELDS, "selected_config_path", "selected_config_sha256",
                "log_device_token", "log_path", "git_log_policy"):
        print(f"[run] {key}={_text(metadata.get(key))}")


if __name__ == "__main__":
    main()
