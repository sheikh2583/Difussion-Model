"""
Generate (noise, image) pairs for Reflow training.

Must be run BEFORE: python train.py --algorithm reflow

This script:
  1. Loads a trained FlowMatchingAlgorithm checkpoint.
  2. Samples z_1 ~ N(0,I) noise.
  3. Runs full Euler integration (high NFE) to get x_0_hat.
  4. Saves {z1, x0} as a .pt file for ReflowAlgorithm to load.

Usage:
    python scripts/generate_reflow_pairs.py \\
        --checkpoint results/fm_<dataset>/checkpoints/run_1/FlowMatchingAlgorithm_epoch100.pt \\
        --config config/<fm-preset>.json \\
        --n-pairs 50000 \\
        --nfe 50 \\
        --output data/reflow_pairs_<dataset>.pt

Runtime depends on the selected device and model configuration.
"""
import argparse
import os
import sys
from contextlib import contextmanager

# Make project packages importable when this file is launched directly as
# `python scripts/generate_reflow_pairs.py` from any working directory.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import torch

from config.config import ExperimentConfig
from models.backbone import build_backbone
from utils.checkpoints import extract_model_state
from utils.device import resolve_device


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config",     required=True)
    p.add_argument("--n-pairs",    type=int, default=50000)
    p.add_argument("--nfe",        type=int, default=50,
                   help="Euler steps for generating clean images. Higher=better quality.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--output",     required=True)
    p.add_argument(
        "--lock-file",
        default="results/.lock",
        help="Shared GPU lock path (default: results/.lock).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output file. Without this flag, fail safely.",
    )
    return p.parse_args()


@contextmanager
def gpu_lock(path: str):
    """Acquire the project-wide GPU lock atomically and release our own lock."""
    lock_path = os.path.abspath(path)
    token = f"pid={os.getpid()}\ncommand=generate_reflow_pairs.py\n"
    os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(
            f"GPU lock already exists: {lock_path}\n"
            "Another project job may be active. Wait for it to finish, or remove "
            "the lock only after confirming it is stale."
        ) from exc

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(token)
        yield
    finally:
        try:
            with open(lock_path, "r", encoding="utf-8") as handle:
                still_ours = handle.read() == token
            if still_ours:
                os.remove(lock_path)
        except FileNotFoundError:
            pass


def _generate(args):
    if args.n_pairs < 1:
        raise ValueError(f"--n-pairs must be >= 1, got {args.n_pairs}")
    if args.nfe < 1:
        raise ValueError(f"--nfe must be >= 1, got {args.nfe}")
    if args.batch_size < 1:
        raise ValueError(f"--batch-size must be >= 1, got {args.batch_size}")
    cfg    = ExperimentConfig.load(args.config)
    device = resolve_device(cfg)

    # Load FM model
    model = build_backbone(cfg.backbone, image_size=cfg.dataset.image_size)
    # Project checkpoints are trusted local artifacts and may contain the
    # serialized ExperimentConfig in addition to tensor state.
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(extract_model_state(state))
    model.to(device)
    model.eval()

    C = cfg.backbone.in_channels
    H = W = cfg.dataset.image_size

    # Preallocate the final CPU tensors. Keeping per-batch lists and then
    # concatenating them temporarily doubled host RAM (about 9.2 GiB at 50k
    # 64x64 RGB pairs), which made CelebA generation unnecessarily fragile.
    z1_all = torch.empty(args.n_pairs, C, H, W, dtype=torch.float32)
    x0_all = torch.empty_like(z1_all)
    n_generated = 0

    print(f"Generating {args.n_pairs} pairs at NFE={args.nfe}...", flush=True)
    with torch.no_grad():
        while n_generated < args.n_pairs:
            bs = min(args.batch_size, args.n_pairs - n_generated)
            z1 = torch.randn(bs, C, H, W, device=device)

            # Full Euler integration: noise → image
            x = z1.clone()
            step = 1.0 / args.nfe
            for i in range(args.nfe):
                t_cur = 1.0 - i * step
                t_b   = torch.full((bs,), t_cur, device=device, dtype=torch.float32)
                v     = model(x, t_b)
                x     = x - v * step
            x0_hat = x.clamp(-1.0, 1.0)

            next_generated = n_generated + bs
            z1_all[n_generated:next_generated].copy_(z1)
            x0_all[n_generated:next_generated].copy_(x0_hat)
            n_generated += bs

            if n_generated % 5000 == 0 or n_generated >= args.n_pairs:
                print(
                    f"  {n_generated}/{args.n_pairs} pairs generated",
                    flush=True,
                )

    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    temporary_path = f"{output_path}.tmp.{os.getpid()}"
    try:
        torch.save({"z1": z1_all, "x0": x0_all}, temporary_path)
        os.replace(temporary_path, output_path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
    print(f"Saved {args.n_pairs} pairs -> {output_path}")
    print(f"Shapes: z1={z1_all.shape}, x0={x0_all.shape}")


def main():
    args = parse_args()
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not os.path.isfile(args.config):
        raise FileNotFoundError(f"Config not found: {args.config}")
    if os.path.exists(args.output) and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {args.output}\n"
            "Pass --overwrite only if replacing it is intentional."
        )
    with gpu_lock(args.lock_file):
        _generate(args)


if __name__ == "__main__":
    main()
