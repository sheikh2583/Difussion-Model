#!/usr/bin/env python3
"""Build a compact, external-agent-friendly thesis context ZIP.

Includes source, configs, tests, report material, metrics, logs, provenance,
plots, and sample images. Excludes datasets, environments, raw checkpoint
tensors, checkpoint archives, and global FID caches.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from verify_project_layout import active_training_processes


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "exports" / "thesis_context.zip"
EXCLUDED_PARTS = {".git", "venv", "__pycache__"}
EXCLUDED_SUFFIXES = {".pt", ".zip", ".npz"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--allow-running",
        action="store_true",
        help="Permit a read-only snapshot while training is active.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def source_files(root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return [
        root / name.decode("utf-8")
        for name in result.stdout.split(b"\0")
        if name and (root / name.decode("utf-8")).is_file()
    ]


def useful_result_files(root: Path) -> list[Path]:
    results = root / "results"
    selected = []
    if not results.is_dir():
        return selected
    for path in results.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(results)
        if relative.parts and relative.parts[0] in {"exports", "metrics"}:
            continue
        if "checkpoints" in relative.parts or path.suffix.lower() in EXCLUDED_SUFFIXES:
            continue
        selected.append(path)
    return selected


def context_files(root: Path) -> list[Path]:
    files = set(source_files(root))
    files.update(useful_result_files(root))
    logs = root / "training_logs"
    if logs.is_dir():
        files.update(path for path in logs.rglob("*") if path.is_file())
    return sorted(
        path
        for path in files
        if path.is_file()
        and not EXCLUDED_PARTS.intersection(path.relative_to(root).parts)
        and path.suffix.lower() not in EXCLUDED_SUFFIXES
    )


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> int:
    args = parse_args()
    root = PROJECT_ROOT
    output = args.output.expanduser().resolve()
    training = active_training_processes(root)
    files = context_files(root)
    total = sum(path.stat().st_size for path in files)

    print(f"Files: {len(files)}")
    print(f"Uncompressed size: {total / 1024**2:.1f} MiB")
    print(f"Output: {output}")
    if training:
        print(f"Active training-related processes: {len(training)}")
    if args.dry_run:
        for path in files:
            print(path.relative_to(root).as_posix())
        return 0
    if training and not args.allow_running:
        print(
            "Training is active; the thesis context ZIP was not changed.\n"
            "Run this after training, or add --allow-running for a read-only snapshot.",
            file=os.sys.stderr,
        )
        return 3

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_files = []
    with tempfile.NamedTemporaryFile(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent, delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with zipfile.ZipFile(
            temporary_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as archive:
            for path in files:
                relative = path.relative_to(root).as_posix()
                archive.write(path, relative)
                manifest_files.append(
                    {"path": relative, "bytes": path.stat().st_size, "sha256": digest(path)}
                )
            manifest = {
                "format": "diffusion-project-thesis-context-v1",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "training_active_during_snapshot": bool(training),
                "exclusions": [
                    "raw datasets",
                    "virtual environments",
                    "raw .pt checkpoints",
                    "checkpoint/bundle ZIPs",
                    "global FID .npz caches",
                ],
                "files": manifest_files,
            }
            archive.writestr("CONTEXT_MANIFEST.json", json.dumps(manifest, indent=2))
            archive.writestr(
                "CONTEXT_README.md",
                "# Thesis context snapshot\n\n"
                "This compact archive is intended for code review and thesis discussion. "
                "It contains source, configuration, tests, report material, run provenance, "
                "metrics, plots, sample images, and training transcripts. Large model and "
                "dataset binaries are deliberately excluded. See `CONTEXT_MANIFEST.json` "
                "for the complete inventory and SHA-256 hashes.\n",
            )
        with zipfile.ZipFile(temporary_path) as archive:
            bad_member = archive.testzip()
            if bad_member:
                raise OSError(f"Corrupt archive member: {bad_member}")
        os.replace(temporary_path, output)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    print(f"Created: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
