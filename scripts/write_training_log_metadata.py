#!/usr/bin/env python3
"""Write a deterministic metadata sidecar for one completed training log."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


RUN_FIELD = re.compile(r"^\[run\]\s+([A-Za-z0-9_.-]+)=(.*)$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()

    log_path = args.log.resolve()
    if not log_path.is_file():
        raise SystemExit(f"Training log does not exist: {log_path}")

    metadata: dict[str, object] = {}
    with log_path.open(encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            match = RUN_FIELD.match(raw_line.rstrip("\n"))
            if match:
                metadata[match.group(1)] = match.group(2)

    metadata.update({
        "metadata_schema_version": 2,
        "transcript_bytes": log_path.stat().st_size,
        "transcript_sha256": sha256(log_path),
    })
    sidecar = log_path.with_suffix(log_path.suffix + ".meta.json")
    sidecar.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(sidecar)


if __name__ == "__main__":
    main()
