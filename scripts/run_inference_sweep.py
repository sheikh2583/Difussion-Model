"""Compatibility entry point for the current checkpoint sample generator.

Prefer ``scripts/generate_checkpoint_samples.py`` directly. This wrapper keeps
the older ``--all`` and singular ``--seed`` command shape while writing
provenance-validated artifacts under ``results/checkpoint_samples/<run>/``.
It does not calculate FID/IS; use ``evaluate.py`` for quantitative metrics.
"""

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATOR = PROJECT_ROOT / "scripts" / "generate_checkpoint_samples.py"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", type=str)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--nfe", type=str, default="1,5,20")
    parser.add_argument("--epochs", type=str)
    parser.add_argument("--n-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if bool(args.experiments) == args.all:
        raise SystemExit("Choose exactly one of --experiments or --all")
    if args.seed < 0:
        raise SystemExit("--seed must be non-negative")

    command = [
        sys.executable,
        str(GENERATOR),
        "--nfe", args.nfe,
        "--n-samples", str(args.n_samples),
        "--seeds", str(args.seed),
    ]
    if args.experiments:
        command.extend(("--experiments", args.experiments))
    if args.epochs:
        command.extend(("--epochs", args.epochs))
    if args.force:
        command.append("--force")
    return subprocess.run(command, cwd=PROJECT_ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
