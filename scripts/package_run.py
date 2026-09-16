#!/usr/bin/env python3
"""Package a completed experiment into one portable ZIP archive."""

from __future__ import annotations

import argparse
import json
import os
import zipfile
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Completed results/run directory.")
    parser.add_argument(
        "--output",
        help="Output ZIP. Defaults to results/exports/<run-name>.zip.",
    )
    return parser.parse_args()


def package_run(run_dir: Path, output: Path) -> Path:
    run_dir = run_dir.resolve()
    output = output.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    checkpoint_archives = sorted((run_dir / "checkpoints" / "archive").glob("*.zip"))
    if not checkpoint_archives:
        raise FileNotFoundError(
            f"No checkpoint archives found in {run_dir / 'checkpoints' / 'archive'}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()

    manifest = {
        "run": run_dir.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_archives": [path.name for path in checkpoint_archives],
        "note": "Raw .pt files are omitted because each checkpoint archive already contains checkpoint.pt.",
    }

    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
            archive.writestr("bundle_manifest.json", json.dumps(manifest, indent=2))
            for path in sorted(run_dir.rglob("*")):
                if not path.is_file():
                    continue
                if output == path.resolve() or temporary == path.resolve():
                    continue
                relative = path.relative_to(run_dir)
                if relative.parts and relative.parts[0] == "exports":
                    continue
                if path.suffix == ".pt" and relative.parts[:1] == ("checkpoints",):
                    continue
                archive.write(path, arcname=str(Path(run_dir.name) / relative))
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    return output


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    output = (
        Path(args.output)
        if args.output
        else PROJECT_ROOT / "results" / "exports" / f"{run_dir.name}.zip"
    )
    packaged = package_run(run_dir, output)
    print(f"Run archive: {packaged}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
