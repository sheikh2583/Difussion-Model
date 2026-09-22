#!/usr/bin/env python3
"""Write a run-local latent config bound to one exact FM teacher checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", required=True)
    args = parser.parse_args()

    raw = json.loads(args.source.read_text(encoding="utf-8"))
    algorithm_kwargs = raw.get("algorithm_kwargs")
    if not isinstance(algorithm_kwargs, dict) or "teacher_checkpoint" not in algorithm_kwargs:
        raise SystemExit(
            f"{args.source} does not define algorithm_kwargs.teacher_checkpoint"
        )
    algorithm_kwargs["teacher_checkpoint"] = args.teacher_checkpoint
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
