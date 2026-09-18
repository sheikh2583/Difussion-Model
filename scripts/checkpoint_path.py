#!/usr/bin/env python3
"""Print one checkpoint path from the latest numbered run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.checkpoint_runs import (
    checkpoint_run_directory,
    find_checkpoint,
    latest_checkpoint_run_number,
    next_checkpoint_run_number,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--class-name", required=True)
    parser.add_argument("--epoch", required=True, type=int)
    parser.add_argument("--run-number", type=int)
    parser.add_argument("--planned-mode", choices=("continue", "fresh"))
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    if args.planned_mode == "fresh":
        number = next_checkpoint_run_number(run_dir)
        path = checkpoint_run_directory(run_dir, number) / (
            f"{args.class_name}_epoch{args.epoch}.pt"
        )
    else:
        path = find_checkpoint(run_dir, args.class_name, args.epoch, args.run_number)
        if path is None and args.planned_mode == "continue":
            number = latest_checkpoint_run_number(run_dir)
            if number is None:
                number = next_checkpoint_run_number(run_dir)
            path = checkpoint_run_directory(run_dir, number) / (
                f"{args.class_name}_epoch{args.epoch}.pt"
            )
    if path is None:
        raise SystemExit(1)
    try:
        print(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        print(path.resolve())


if __name__ == "__main__":
    main()
