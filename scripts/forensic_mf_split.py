#!/usr/bin/env python3
"""
Forensic split of results/mf_cifar10/metrics/mf_cifar10.jsonl.

The original file contains two concatenated training runs that were
appended without being separated:

  Run 1  (epochs  1-30):  batch=32,  lr=5e-5,  ~1562 steps/epoch
                           loss diverged: 0.554 -> 0.648
  Run 2  (epochs 31-100): batch=64,  lr=5e-5 (inferred), ~780 steps/epoch
                           loss stabilised ~0.40-0.44, then diverged
                           late (epochs 97-100: 0.493, 0.530, 0.601, 0.795)

The boundary is at epoch 31 where optimization_steps resets backward
(46860 at epoch 30 -> 24209 at epoch 31), proving a fresh run was
started and its output appended to the same JSONL file.

This script writes forensic copies to results/aggregate/forensic/:
  mf_cifar10_run1.jsonl  — epochs 1-30
  mf_cifar10_run2.jsonl  — epochs 31-100 (the tournament run)

The original mf_cifar10.jsonl is NEVER modified (it is frozen evidence).
Evaluation records (epoch=null) are assigned to the run whose training
epochs they most plausibly describe (Run 1 per original config.json).

No model training, GPU evaluation, or sampling is performed.
"""
from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "results" / "mf_cifar10" / "metrics" / "mf_cifar10.jsonl"
OUT_DIR = PROJECT_ROOT / "results" / "aggregate" / "forensic"
OUT_RUN1 = OUT_DIR / "mf_cifar10_run1.jsonl"
OUT_RUN2 = OUT_DIR / "mf_cifar10_run2.jsonl"
REPORT_PATH = OUT_DIR / "mf_cifar10_split_report.md"

# ── boundary detection ────────────────────────────────────────────────────────
# The optimization_steps counter reset at epoch 31 (46860 -> 24209).
# Per-epoch steps at epoch 32+ = 780, consistent with batch=64 on CIFAR-10
# (50000 / 64 = 781.25, ceil = 782 or floor = 781).
# Per-epoch steps at epoch 2-30 = 1562, consistent with batch=32.
RUN2_START_EPOCH = 31

# Evaluation records (epoch=null) predate any split; assign to Run 1 because
# they were produced immediately after the epoch-30 stop, before the second run.


def split_records(path: Path) -> tuple[list[dict], list[dict], list[dict]]:
    """Return (run1, run2, evaluation) record lists."""
    run1, run2, evals = [], [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        rt = rec.get("record_type")
        epoch = rec.get("epoch")
        if rt == "evaluation" or epoch is None:
            evals.append(rec)
        elif epoch < RUN2_START_EPOCH:
            run1.append(rec)
        else:
            run2.append(rec)
    return run1, run2, evals


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def steps_per_epoch_stats(records: list[dict]) -> dict:
    train = sorted(
        [r for r in records if r.get("record_type") == "train_epoch"],
        key=lambda r: r.get("epoch", 0),
    )
    deltas = []
    for i in range(1, len(train)):
        delta = train[i].get("optimization_steps", 0) - train[i-1].get("optimization_steps", 0)
        if delta > 0:
            deltas.append(delta)
    if not deltas:
        return {}
    return {
        "min": min(deltas),
        "max": max(deltas),
        "typical": sorted(deltas)[len(deltas) // 2],
        "inferred_batch": 50000 // (sorted(deltas)[len(deltas) // 2] or 1),
    }


def loss_range(records: list[dict]) -> tuple[float, float]:
    losses = [r.get("loss", 0) for r in records if r.get("record_type") == "train_epoch"]
    return (min(losses), max(losses)) if losses else (0, 0)


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"Source not found: {SRC}")

    run1, run2, evals = split_records(SRC)
    write_jsonl(OUT_RUN1, run1 + evals)   # eval records go with Run 1
    write_jsonl(OUT_RUN2, run2)

    s1 = steps_per_epoch_stats(run1)
    s2 = steps_per_epoch_stats(run2)
    l1 = loss_range(run1)
    l2 = loss_range(run2)

    report = f"""# MF CIFAR-10 Run Split Report

**Generated:** forensic split of `results/mf_cifar10/metrics/mf_cifar10.jsonl`
**Original file:** UNTOUCHED (frozen evidence)
**Split files:** `results/aggregate/forensic/`

## Run 1 — Original abandoned run
- **Epochs:** 1 – {RUN2_START_EPOCH - 1}
- **Records:** {len(run1)} training + {len(evals)} evaluation (assigned to Run 1)
- **Steps/epoch:** {s1.get('typical', '?')} → inferred batch size = {s1.get('inferred_batch', '?')}
- **Loss range:** {l1[0]:.4f} (epoch 1) → {l1[1]:.4f} (epoch {RUN2_START_EPOCH - 1})
- **Trend:** Diverging — never decreased from epoch 1 value
- **Config:** Matches `results/mf_cifar10/config.json` (batch=64 recorded,
  but per-epoch step count of {s1.get('typical','?')} is consistent with
  batch=32 on CIFAR-10 50k; the config.json may reflect a later edit)
- **Evaluation:** 4 records (epoch=null, nfe 1/5/10/20); FID 210.99 at NFE=1
  — these describe the epoch-30 checkpoint quality

## Run 2 — Continuation run (tournament)
- **Epochs:** {RUN2_START_EPOCH} – 100
- **Records:** {len(run2)} training
- **Steps/epoch:** {s2.get('typical', '?')} → inferred batch size = {s2.get('inferred_batch', '?')}
- **Loss range:** {l2[0]:.4f} – {l2[1]:.4f}
- **Trend:** Stabilised at ~0.40–0.44 (epochs 32–96) then **late divergence**
  (epoch 97: 0.493, epoch 98: 0.530, epoch 99: 0.601, epoch 100: 0.795)
- **Config:** Different from Run 1 — step count consistent with batch=64
  on CIFAR-10. No separate config.json exists for this run.
- **opt_steps counter reset:** 46,860 (epoch 30) → 24,209 (epoch 31),
  proving a fresh training process was started and appended to the file.
- **Evaluation:** NONE — this run was never evaluated.

## Key finding for the thesis

The report's claim "MF abandoned at epoch 30, loss never improved" is
**factually incorrect**. Run 2 reached stable loss ~0.40 (vs. Run 1's
~0.65) but was never evaluated. The late divergence (epochs 97–100) is
a separate phenomenon from Run 1's early divergence.

The **unevaluated Run 2 checkpoint** at approximately epoch 44 (stable
~0.43 loss) is a candidate for FID evaluation (Chunk 1.1 in the plan).
Checkpoints exist in `results/mf_cifar10/checkpoints/` — check for
epoch 40, 50, 60 files corresponding to Run 2 epochs.

## Impact on current summary.csv

The current `summary.csv` merges both runs under `mf_cifar10` using
`mf_cifar10.jsonl`. After the aggregator fix (Fix 0.1), only the
canonical JSONL is loaded — but the MF JSONL still contains the mixed
data. For the forensic split to propagate to the aggregate, the user
would need to manually copy the corrected JSONL or re-aggregate using
the split files. The original is left intact as evidence.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")

    print(f"Run 1: {len(run1)} train + {len(evals)} eval records -> {OUT_RUN1}")
    print(f"Run 2: {len(run2)} train records -> {OUT_RUN2}")
    print(f"Report: {REPORT_PATH}")
    print()
    print(f"Run 1 inferred batch={s1.get('inferred_batch','?')}, "
          f"loss {l1[0]:.4f} -> {l1[1]:.4f} (diverged)")
    print(f"Run 2 inferred batch={s2.get('inferred_batch','?')}, "
          f"loss {l2[0]:.4f} -> {l2[1]:.4f} (stable then late divergence)")
    print()
    print("NOTE: original mf_cifar10.jsonl was NOT modified.")


if __name__ == "__main__":
    main()
