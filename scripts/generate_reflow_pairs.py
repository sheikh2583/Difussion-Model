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
        --checkpoint results/fm_cifar10/checkpoints/FlowMatchingAlgorithm_epoch100.pt \\
        --config config/fm_full.json \\
        --n-pairs 50000 \\
        --nfe 50 \\
        --output data/reflow_pairs_cifar10.pt

    python scripts/generate_reflow_pairs.py \\
        --checkpoint results/fm_celeba64/checkpoints/FlowMatchingAlgorithm_epoch100.pt \\
        --config config/fm_celeba64.json \\
        --n-pairs 50000 \\
        --nfe 50 \\
        --output data/reflow_pairs_celeba64.pt

Time estimate: ~30min for 50k pairs at NFE=50 on RTX 3060.
"""
import argparse
import os
import sys

# Make project packages importable when this file is launched directly as
# `python scripts/generate_reflow_pairs.py` from any working directory.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import torch

from algorithms.flow_matching import FlowMatchingAlgorithm
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
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--output",     required=True)
    return p.parse_args()


def main():
    args = parse_args()
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

    algorithm = FlowMatchingAlgorithm(model)

    C = cfg.backbone.in_channels
    H = W = cfg.dataset.image_size

    all_z1 = []
    all_x0 = []
    n_generated = 0

    print(f"Generating {args.n_pairs} pairs at NFE={args.nfe}...")
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

            all_z1.append(z1.cpu())
            all_x0.append(x0_hat.cpu())
            n_generated += bs

            if n_generated % 5000 == 0 or n_generated >= args.n_pairs:
                print(f"  {n_generated}/{args.n_pairs} pairs generated")

    z1_all = torch.cat(all_z1, dim=0)[:args.n_pairs]
    x0_all = torch.cat(all_x0, dim=0)[:args.n_pairs]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save({"z1": z1_all, "x0": x0_all}, args.output)
    print(f"Saved {args.n_pairs} pairs -> {args.output}")
    print(f"Shapes: z1={z1_all.shape}, x0={x0_all.shape}")


if __name__ == "__main__":
    main()
