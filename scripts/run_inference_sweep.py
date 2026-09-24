"""
run_inference_sweep.py
----------------------
Generate sample grids from trained checkpoints at multiple NFE values.
This produces the images that the web dashboard uses for noise-to-image
visualization and enables FID/IS computation.

Usage (from project root, venv activated):
    python scripts/run_inference_sweep.py --experiments fm_cifar10,fm_lognorm_cifar10,mf_cifar10
    python scripts/run_inference_sweep.py --all
    python scripts/run_inference_sweep.py --experiments fm_cifar10 --nfe 1,5,10,20,50
    python scripts/run_inference_sweep.py --experiments fm_cifar10 --epochs 50,100

After running, re-run prepare_web_data.py to update the dashboard manifest:
    python scripts/prepare_web_data.py

NOTE: This script requires GPU access and is intended for user execution only.
      Agents must not execute this script per AGENTS.md rules.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torchvision.utils import save_image

from algorithms import ALGORITHM_REGISTRY
from config.config import ExperimentConfig
from models.backbone import build_backbone
from utils.checkpoint_runs import latest_checkpoint_run_directory
from utils.checkpoints import load_algorithm_state

RESULTS_ROOT = Path("results")
OUT_ROOT = Path("results/checkpoint_samples")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Map algo class name → short directory name used in checkpoint_samples/
ALGO_DIR_MAP = {
    "FlowMatchingAlgorithm": "fm",
    "FlowMatchingLognormAlgorithm": "fm_lognorm",
    "MeanFlowAlgorithm": "mf",
    "MeanFlowDistillAlgorithm": "mf_distill",
    "ConsistencyAlgorithm": "consistency",
    "ReflowAlgorithm": "reflow",
}

DEFAULT_NFE = [1, 5, 20]
DEFAULT_N_SAMPLES = 64
DEFAULT_SEED = 0


def infer_algorithm_cls(ckpt_dir: Path):
    """Guess the algorithm class from checkpoint filenames."""
    cls_map = {cls.__name__: cls for cls in ALGORITHM_REGISTRY.values()}
    for fname in sorted(os.listdir(ckpt_dir)):
        if not fname.endswith(".pt"):
            continue
        for cls_name, cls in cls_map.items():
            if fname.startswith(cls_name):
                return cls, cls_name
    return None, None


def find_checkpoints(ckpt_dir: Path, cls_name: str, epochs=None):
    """Find checkpoint .pt files, optionally filtered by epoch list."""
    import re
    pattern = re.compile(rf"^{re.escape(cls_name)}_epoch(\d+)\.pt$")
    found = []
    for fname in sorted(os.listdir(ckpt_dir)):
        m = pattern.match(fname)
        if m:
            epoch = int(m.group(1))
            if epochs is None or epoch in epochs:
                found.append((epoch, ckpt_dir / fname))
    return sorted(found, key=lambda x: x[0])


def load_algorithm(cfg, algorithm_cls, ckpt_path, device):
    """Rebuild model + algorithm from config, then load checkpoint weights."""
    model = build_backbone(cfg.backbone, image_size=cfg.dataset.image_size)
    algorithm = algorithm_cls(model, algorithm_kwargs=cfg.algorithm_kwargs)
    for m in algorithm.trainable_modules():
        m.to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    load_algorithm_state(algorithm, ckpt)
    for m in algorithm.trainable_modules():
        m.eval()
    return algorithm


@torch.no_grad()
def generate_samples(algorithm, nfe, n_samples, device, seed):
    """Generate sample images using the algorithm's sampler."""
    torch.manual_seed(seed)
    return algorithm.sample(n_samples, nfe, device)


def process_experiment(experiment_name, nfe_values, epochs, n_samples, seed, force):
    """Process a single experiment: generate grids for all epoch×NFE combos."""
    run_dir = RESULTS_ROOT / experiment_name
    cfg_path = run_dir / "config.json"

    if not cfg_path.exists():
        print(f"[SKIP] No config.json in {run_dir}")
        return

    active_ckpt_dir = latest_checkpoint_run_directory(run_dir)
    if active_ckpt_dir is None:
        print(f"[SKIP] No checkpoint run found in {run_dir}")
        return

    algo_cls, cls_name = infer_algorithm_cls(active_ckpt_dir)
    if algo_cls is None:
        print(f"[SKIP] Cannot infer algorithm from {active_ckpt_dir}")
        return

    algo_dir_name = ALGO_DIR_MAP.get(cls_name, experiment_name)
    out_dir = OUT_ROOT / algo_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = ExperimentConfig.load(str(cfg_path))
    checkpoints = find_checkpoints(active_ckpt_dir, cls_name, epochs)

    if not checkpoints:
        print(f"[SKIP] No matching checkpoints in {active_ckpt_dir}")
        return

    print(f"\n{'='*60}")
    print(f"Experiment: {experiment_name}")
    print(f"Algorithm:  {cls_name}")
    print(f"Output dir: {out_dir}")
    print(f"Checkpoints: {len(checkpoints)} (epochs {checkpoints[0][0]}-{checkpoints[-1][0]})")
    print(f"NFE values: {nfe_values}")
    print(f"{'='*60}")

    for epoch, ckpt_path in checkpoints:
        print(f"\n  Loading epoch {epoch}: {ckpt_path.name}")
        try:
            algorithm = load_algorithm(cfg, algo_cls, ckpt_path, DEVICE)
        except Exception as e:
            print(f"  [ERROR] Failed to load checkpoint: {e}")
            continue

        for nfe in nfe_values:
            out_path = out_dir / f"epoch{epoch:03d}_nfe{nfe}.png"
            if out_path.exists() and not force:
                print(f"    NFE {nfe:>3}: {out_path.name} (exists, skipping)")
                continue

            print(f"    NFE {nfe:>3}: generating {n_samples} samples...", end=" ", flush=True)
            try:
                samples = generate_samples(algorithm, nfe, n_samples, DEVICE, seed)
                import math
                nrow = max(1, math.ceil(math.sqrt(n_samples)))
                save_image(samples, out_path, nrow=nrow, normalize=True,
                          value_range=(-1, 1), padding=2)
                print(f"✓ saved to {out_path.name}")
            except Exception as e:
                print(f"✗ error: {e}")

        # Free GPU memory
        del algorithm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def discover_all_experiments():
    """Find all valid experiment directories under results/."""
    experiments = []
    for d in sorted(RESULTS_ROOT.iterdir()):
        if not d.is_dir():
            continue
        if (d / "config.json").exists() and (d / "checkpoints").is_dir():
            experiments.append(d.name)
    return experiments


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate inference sample grids for the web dashboard"
    )
    parser.add_argument(
        "--experiments",
        type=str,
        default=None,
        help="Comma-separated experiment names (e.g. fm_cifar10,mf_cifar10)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all discovered experiments",
    )
    parser.add_argument(
        "--nfe",
        type=str,
        default=",".join(str(n) for n in DEFAULT_NFE),
        help=f"Comma-separated NFE values (default: {DEFAULT_NFE})",
    )
    parser.add_argument(
        "--epochs",
        type=str,
        default=None,
        help="Comma-separated epoch numbers to process (default: all checkpoints)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=DEFAULT_N_SAMPLES,
        help=f"Number of samples per grid (default: {DEFAULT_N_SAMPLES})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for sampling (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing sample images",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.all:
        experiments = discover_all_experiments()
    elif args.experiments:
        experiments = [e.strip() for e in args.experiments.split(",")]
    else:
        print("Please specify --experiments or --all")
        sys.exit(1)

    nfe_values = [int(n.strip()) for n in args.nfe.split(",")]
    epochs = None
    if args.epochs:
        epochs = set(int(e.strip()) for e in args.epochs.split(","))

    print(f"Device: {DEVICE}")
    print(f"Experiments: {experiments}")
    print(f"NFE values: {nfe_values}")
    if epochs:
        print(f"Epochs filter: {sorted(epochs)}")

    for experiment in experiments:
        process_experiment(
            experiment,
            nfe_values=nfe_values,
            epochs=epochs,
            n_samples=args.n_samples,
            seed=args.seed,
            force=args.force,
        )

    print(f"\n{'='*60}")
    print("Done! Now run: python scripts/prepare_web_data.py")
    print("to update the dashboard manifest.")


if __name__ == "__main__":
    main()
