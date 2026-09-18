#!/usr/bin/env python3
"""Preview or apply non-destructive migration of flat checkpoints to run_1."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.checkpoint_runs import legacy_migration_plan, migrate_legacy_checkpoint_layout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--apply", action="store_true", help="Move files after collision checks.")
    args = parser.parse_args()
    results = Path(args.results_dir)
    if not results.is_absolute():
        results = PROJECT_ROOT / results
    if not results.is_dir():
        print(f"No results directory found: {results}")
        return

    plans: list[tuple[Path, list[tuple[Path, Path]]]] = []
    for root in sorted(results.rglob("checkpoints")):
        if root.is_dir():
            plan = legacy_migration_plan(root.parent)
            if plan:
                plans.append((root.parent, plan))
    if not plans:
        print("Checkpoint layout is already organized; nothing to migrate.")
        return

    for run_dir, plan in plans:
        print(f"[{run_dir.relative_to(results)}]")
        for source, destination in plan:
            print(f"  {source.relative_to(results)} -> {destination.relative_to(results)}")
    if not args.apply:
        print(f"Preview only: {sum(len(plan) for _, plan in plans)} files; rerun with --apply.")
        return

    moved = 0
    for run_dir, _ in plans:
        moved += len(migrate_legacy_checkpoint_layout(run_dir))
    print(f"Migration complete: moved {moved} files without overwriting any destination.")


if __name__ == "__main__":
    main()
