"""Re-score latent-space checkpoints against the correct raw-pixel FID reference.

Background
----------
Before the evaluator fix, latent experiments built their FID reference from the
*latent* validation loader (5,000 normalized 3x16x16 tensors) and compared it
with decoded 64x64 RGB samples, producing meaningless FID values near 300.
``ensure_fid_reference`` now dispatches latent configs to
``ensure_fid_reference_latent`` (raw CelebA validation pixels).

This script regenerates samples from existing checkpoints with the same
protocol as the original evaluation (seed from the saved config, sampling in
batches of 32, decode in batches of 32) and scores them with the corrected
reference. It NEVER writes into ``results/<experiment>/``; the original
metrics stay untouched for comparison. Outputs go to
``results/aggregate/latent_rescored/``.

Usage
-----
    venv/bin/python scripts/rescore_latent_fid.py            # all experiments
    venv/bin/python scripts/rescore_latent_fid.py --experiments fm_celeba_latent
    venv/bin/python scripts/rescore_latent_fid.py --dry-run
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from algorithms import ALGORITHM_REGISTRY  # noqa: E402
from config.config import ExperimentConfig  # noqa: E402
from evaluation.evaluator import (  # noqa: E402
    _decode_latents_batched,
    _load_codec_checkpoint,
    ensure_fid_reference,
)
from evaluation.metrics import (  # noqa: E402
    compute_fid_from_prepared,
    compute_inception_score,
    load_real_images,
    prepare_fid,
)
from models.backbone import build_backbone  # noqa: E402
from utils.checkpoints import load_algorithm_state  # noqa: E402
from utils.gpu_lock import acquire_gpu_lock  # noqa: E402

SAMPLE_BATCH = 32
CLASS_TO_KEY = {cls.__name__: key for key, cls in ALGORITHM_REGISTRY.items()}
CHECKPOINT_RE = re.compile(r"(?P<cls>\w+)_epoch(?P<epoch>\d+)\.pt$")


def original_records(experiment: str) -> list[dict]:
    rows = []
    for path in (ROOT / "results" / experiment / "metrics").glob("*.jsonl"):
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("record_type") == "evaluation":
                rows.append(row)
    return rows


def resolve(path: str | None) -> Path | None:
    if not path:
        return None
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return candidate if candidate.is_file() else None


def select_checkpoints(experiment: str) -> list[Path]:
    """Originally evaluated checkpoint (if still on disk) + every retained one in its run."""
    originals = original_records(experiment)
    chosen: list[Path] = []
    if originals:
        last_epoch = max(r.get("epoch") or 0 for r in originals)
        for row in originals:
            if (row.get("epoch") or 0) == last_epoch:
                found = resolve(row.get("checkpoint_path"))
                if found and found not in chosen:
                    chosen.append(found)
    pool = list((ROOT / "results" / experiment / "checkpoints").glob("run_*/*_epoch*.pt"))
    if chosen:
        run_dir = chosen[0].parent
        pool = [p for p in pool if p.parent == run_dir]
    else:
        # No readable original evaluation: use the newest numbered run.
        runs = sorted({p.parent for p in pool}, key=lambda p: int(p.name.split("_")[1]))
        pool = [p for p in pool if runs and p.parent == runs[-1]]
    # Every retained checkpoint of the chosen run (a few per experiment), so
    # late-training regressions such as latent MF epoch 80 vs 100 stay visible.
    for path in sorted(pool, key=lambda p: int(CHECKPOINT_RE.search(p.name)["epoch"])):
        if path not in chosen:
            chosen.append(path)
    return chosen


def build_algorithm(cfg: ExperimentConfig, ckpt: Path, device: torch.device):
    key = CLASS_TO_KEY[CHECKPOINT_RE.search(ckpt.name)["cls"]]
    model = build_backbone(cfg.backbone, image_size=cfg.dataset.image_size)
    algo = ALGORITHM_REGISTRY[key](model, cfg.algorithm_kwargs)
    load_algorithm_state(algo, torch.load(ckpt, map_location="cpu", weights_only=False))
    for module in algo.trainable_modules():
        module.to(device).eval()
    return key, algo


@torch.no_grad()
def generate(algo, count: int, nfe: int, seed: int, device: torch.device) -> torch.Tensor:
    batches = []
    devices = [device.index if device.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=devices if device.type == "cuda" else []):
        torch.manual_seed(seed)
        for start in range(0, count, SAMPLE_BATCH):
            batches.append(algo.sample(min(SAMPLE_BATCH, count - start), nfe, device).cpu())
    return torch.cat(batches)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--experiments", nargs="*", default=None,
                        help="Experiment directory names (default: every *_celeba_latent*).")
    parser.add_argument("--output-dir", default="results/aggregate/latent_rescored")
    parser.add_argument("--reference-cache",
                        default="results/metrics/fid_reference_stats_celeba64_raw_for_latent.npz")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    names = args.experiments or sorted(
        p.name for p in (ROOT / "results").glob("*_celeba_latent*")
        if (p / "config.json").is_file() and p.name != "mf_hutchinson_celeba_latent"
    )
    plan = {name: select_checkpoints(name) for name in names}
    for name, ckpts in plan.items():
        print(f"{name}: {[str(c.relative_to(ROOT)) for c in ckpts] or 'NO CHECKPOINT'}")
    if args.dry_run:
        return 0

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / "rescored_latent_fid.jsonl"
    done = set()
    if jsonl.is_file():
        for line in jsonl.read_text().splitlines():
            row = json.loads(line)
            done.add((row["checkpoint"], row["nfe"]))

    device = torch.device(args.device)
    with acquire_gpu_lock(command="rescore_latent_fid.py"):
        # Build the corrected reference through the FIXED evaluator entry point.
        first_cfg = ExperimentConfig.load(ROOT / "results" / names[0] / "config.json")
        first_cfg.evaluation.fid_reference_cache = str(ROOT / args.reference_cache)
        ref_path = ensure_fid_reference(first_cfg, None, device)
        reference = load_real_images(ref_path, torch.device("cpu"))
        print(f"reference: {tuple(reference.shape)} range=[{reference.min():.2f},{reference.max():.2f}]")
        fid_metric = prepare_fid(reference, device)
        codec = None

        for name, ckpts in plan.items():
            cfg = ExperimentConfig.load(ROOT / "results" / name / "config.json")
            originals = original_records(name)
            if codec is None:
                codec = _load_codec_checkpoint(str(ROOT / cfg.dataset.codec_checkpoint), device)
            for ckpt in ckpts:
                epoch = int(CHECKPOINT_RE.search(ckpt.name)["epoch"])
                key, algo = build_algorithm(cfg, ckpt, device)
                for nfe in cfg.evaluation.nfe_values:
                    rel = str(ckpt.relative_to(ROOT))
                    if (rel, nfe) in done:
                        continue
                    t0 = time.time()
                    latents = generate(algo, cfg.evaluation.num_generated_samples, nfe,
                                       cfg.seed, device)
                    t1 = time.time()
                    images = _decode_latents_batched(codec, latents)
                    t2 = time.time()
                    fid = compute_fid_from_prepared(fid_metric, images, device)
                    is_mean, is_std = compute_inception_score(images, device)
                    old = next((r.get("fid") for r in originals
                                if r.get("epoch") == epoch and r.get("nfe") == nfe), None)
                    row = {
                        "experiment": name, "algorithm_key": key, "checkpoint": rel,
                        "epoch": epoch, "nfe": nfe, "num_samples": len(images),
                        "fid_corrected": fid, "is_mean": is_mean, "is_std": is_std,
                        "fid_original_invalid": old,
                        "backbone_seconds": t1 - t0, "decode_seconds": t2 - t1,
                        "reference": args.reference_cache,
                    }
                    with jsonl.open("a") as handle:
                        handle.write(json.dumps(row) + "\n")
                    print(f"{name} ep{epoch} nfe={nfe}: FID {fid:.3f} (old {old}) IS {is_mean:.3f}",
                          flush=True)
                del algo
                torch.cuda.empty_cache()

    rows = [json.loads(line) for line in jsonl.read_text().splitlines()]
    with (out_dir / "rescored_latent_fid.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
