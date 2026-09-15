"""
generate_checkpoints_4070.py
----------------------------
Generates sample image grids from every saved checkpoint for both
Flow Matching (FM) and Mean Flow (MF) on the 4070 laptop.

Saves one image grid per checkpoint per NFE value, so you can see
how generation quality evolves from epoch 10 to epoch 100 (FM)
and epoch 10 to epoch 25 (MF).

Usage (from project root, venv activated):
    python generate_checkpoints_4070.py

Output:
    results/checkpoint_samples/fm/epoch10_nfe1.png
    results/checkpoint_samples/fm/epoch10_nfe20.png
    ...
    results/checkpoint_samples/mf/epoch10_nfe1.png
    ...
"""

import os
import torch
from torchvision.utils import save_image

from config.config import ExperimentConfig
from models.backbone import build_backbone
from algorithms.flow_matching import FlowMatchingAlgorithm
from algorithms.mean_flow import MeanFlowAlgorithm

# ------------------------------------------------------------------ #
# SETTINGS — adjust if needed
# ------------------------------------------------------------------ #
N_SAMPLES   = 64          # images per grid (8x8)
NFE_VALUES  = [1, 5, 20]  # which NFE values to sample at
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED        = 0
OUT_ROOT    = "./results/checkpoint_samples"

RUNS = [
    {
        "label":         "fm",
        "config_path":   "./results/fm_cifar10/config.json",
        "ckpt_dir":      "./results/fm_cifar10/checkpoints",
        "algorithm_cls": FlowMatchingAlgorithm,
        "ckpt_prefix":   "FlowMatchingAlgorithm_epoch",
    },
    {
        "label":         "mf",
        "config_path":   "./results/mf_cifar10/config.json",
        "ckpt_dir":      "./results/mf_cifar10/checkpoints",
        "algorithm_cls": MeanFlowAlgorithm,
        "ckpt_prefix":   "MeanFlowAlgorithm_epoch",
    },
]
# ------------------------------------------------------------------ #


def load_algorithm(cfg, algorithm_cls, ckpt_path, device):
    """Rebuild model + algorithm from config, load checkpoint weights."""
    model = build_backbone(cfg.backbone, image_size=cfg.dataset.image_size)
    algorithm = algorithm_cls(model, algorithm_kwargs=cfg.algorithm_kwargs)

    # Move all trainable modules to device first
    for m in algorithm.trainable_modules():
        m.to(device)

    # Load checkpoint — weights_only=False needed for custom config object
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    for module, state_dict in zip(
        algorithm.trainable_modules(), ckpt["module_state_dicts"]
    ):
        module.load_state_dict(state_dict)

    # Switch to eval mode
    for m in algorithm.trainable_modules():
        m.eval()

    epoch = ckpt["epoch"]
    return algorithm, epoch


def generate_grid(algorithm, nfe, n_samples, device, seed):
    """Run algorithm.sample() and return image tensor."""
    torch.manual_seed(seed)
    with torch.no_grad():
        images = algorithm.sample(n_samples, nfe, device)
    return images  # (N, C, H, W) in [-1, 1]


def main():
    for run in RUNS:
        label       = run["label"]
        config_path = run["config_path"]
        ckpt_dir    = run["ckpt_dir"]
        algo_cls    = run["algorithm_cls"]
        prefix      = run["ckpt_prefix"]

        out_dir = os.path.join(OUT_ROOT, label)
        os.makedirs(out_dir, exist_ok=True)

        if not os.path.exists(config_path):
            print(f"[SKIP] Config not found: {config_path}")
            continue

        cfg = ExperimentConfig.load(config_path)

        # Find all checkpoint files for this run
        ckpt_files = sorted([
            f for f in os.listdir(ckpt_dir)
            if f.startswith(prefix) and f.endswith(".pt")
        ])

        if not ckpt_files:
            print(f"[SKIP] No checkpoints found in {ckpt_dir}")
            continue

        print(f"\n{'='*60}")
        print(f"  {label.upper()} — {len(ckpt_files)} checkpoints found")
        print(f"{'='*60}")

        for ckpt_file in ckpt_files:
            ckpt_path = os.path.join(ckpt_dir, ckpt_file)

            try:
                algorithm, epoch = load_algorithm(
                    cfg, algo_cls, ckpt_path, DEVICE
                )
            except Exception as e:
                print(f"  [ERROR] Could not load {ckpt_file}: {e}")
                continue

            for nfe in NFE_VALUES:
                out_path = os.path.join(out_dir, f"epoch{epoch:03d}_nfe{nfe}.png")

                if os.path.exists(out_path):
                    print(f"  [SKIP] Already exists: {out_path}")
                    continue

                try:
                    images = generate_grid(algorithm, nfe, N_SAMPLES, DEVICE, SEED)
                    save_image(
                        images,
                        out_path,
                        nrow=8,
                        normalize=True,
                        value_range=(-1, 1),
                    )
                    print(f"  [OK] epoch={epoch:03d} nfe={nfe:3d} -> {out_path}")
                except Exception as e:
                    print(f"  [ERROR] epoch={epoch} nfe={nfe}: {e}")

    print(f"\nDone. Images saved to {OUT_ROOT}/")


if __name__ == "__main__":
    main()