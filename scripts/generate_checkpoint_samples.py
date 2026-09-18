"""
generate_checkpoint_samples.py
-------------------------------
Generate image grids from every saved checkpoint for one or more
experiment runs.  Works with any registered algorithm; no
hardware-specific settings.

Usage (run from project root, venv activated):
    # Auto-discover all runs under results/:
    python scripts/generate_checkpoint_samples.py

    # Specific experiments:
    python scripts/generate_checkpoint_samples.py --experiments fm_cifar10,mf_cifar10

    # Custom NFE values and sample count:
    python scripts/generate_checkpoint_samples.py --nfe 1,5,20 --n-samples 64

    # Custom output root:
    python scripts/generate_checkpoint_samples.py --out-dir ./my_samples

Output:
    results/checkpoint_samples/<experiment>/epoch<NNN>_nfe<M>.png
    (one 8×8 grid per checkpoint per NFE value)
"""

import argparse
import os
import sys

# Allow running from scripts/ subdirectory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torchvision.utils import save_image

from algorithms import ALGORITHM_REGISTRY
from config.config import ExperimentConfig
from models.backbone import build_backbone
from utils.checkpoints import load_algorithm_state
from utils.checkpoint_runs import latest_checkpoint_run_directory

RESULTS_ROOT = "./results"
OUT_ROOT     = "./results/checkpoint_samples"
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED         = 0


# ---------------------------------------------------------------------------
# Algorithm class lookup by checkpoint class-name prefix
# ---------------------------------------------------------------------------
def _build_cls_prefix_map():
    """Map checkpoint filename prefix → algorithm class."""
    mapping = {}
    for algo_cls in ALGORITHM_REGISTRY.values():
        # Instantiate briefly to get the class name used in checkpoint files
        name = algo_cls.__name__          # e.g. "FlowMatchingAlgorithm"
        mapping[name] = algo_cls
    return mapping


def discover_runs(results_root: str):
    """
    Auto-discover all experiment runs under results_root.
    A valid run directory must have a config.json and a checkpoints/ sub-dir.
    Returns list of experiment names.
    """
    runs = []
    if not os.path.isdir(results_root):
        return runs
    for name in sorted(os.listdir(results_root)):
        run_dir = os.path.join(results_root, name)
        if (os.path.isdir(run_dir)
                and os.path.exists(os.path.join(run_dir, "config.json"))
                and os.path.isdir(os.path.join(run_dir, "checkpoints"))):
            runs.append(name)
    return runs


def load_algorithm(cfg, algorithm_cls, ckpt_path, device):
    """Rebuild model + algorithm from config, then load checkpoint weights."""
    model     = build_backbone(cfg.backbone, image_size=cfg.dataset.image_size)
    algorithm = algorithm_cls(model, algorithm_kwargs=cfg.algorithm_kwargs)

    for m in algorithm.trainable_modules():
        m.to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    load_algorithm_state(algorithm, ckpt)
    epoch = ckpt.get("epoch", 0)

    for m in algorithm.trainable_modules():
        m.eval()

    return algorithm, epoch


def generate_grid(algorithm, nfe, n_samples, device, seed):
    """Run algorithm.sample() and return image tensor (N, C, H, W) in [-1,1]."""
    torch.manual_seed(seed)
    with torch.no_grad():
        images = algorithm.sample(n_samples, nfe, device)
    return images


def infer_algorithm_cls(ckpt_dir: str):
    """
    Guess the algorithm class from checkpoint filenames.
    e.g. 'FlowMatchingAlgorithm_epoch10.pt' → FlowMatchingAlgorithm
    """
    cls_map = _build_cls_prefix_map()
    for fname in os.listdir(ckpt_dir):
        if not fname.endswith(".pt"):
            continue
        for cls_name, cls in cls_map.items():
            if fname.startswith(cls_name):
                return cls, cls_name
    return None, None


def process_run(experiment_name, results_root, out_root, nfe_values, n_samples, device):
    run_dir  = os.path.join(results_root, experiment_name)
    cfg_path = os.path.join(run_dir, "config.json")
    checkpoint_root = os.path.join(run_dir, "checkpoints")
    out_dir  = os.path.join(out_root, experiment_name)

    if not os.path.exists(cfg_path):
        print(f"[SKIP] No config.json in {run_dir}")
        return
    if not os.path.isdir(checkpoint_root):
        print(f"[SKIP] No checkpoints/ dir in {run_dir}")
        return

    active_checkpoint_dir = latest_checkpoint_run_directory(Path(run_dir))
    if active_checkpoint_dir is None:
        print(f"[SKIP] No checkpoint run found in {checkpoint_root}")
        return
    ckpt_dir = str(active_checkpoint_dir)

    algo_cls, cls_name = infer_algorithm_cls(ckpt_dir)
    if algo_cls is None:
        print(f"[SKIP] Could not infer algorithm class from checkpoints in {ckpt_dir}")
        return

    cfg = ExperimentConfig.load(cfg_path)
    ckpt_files = sorted(
        f for f in os.listdir(ckpt_dir)
        if f.startswith(cls_name) and f.endswith(".pt")
    )

    if not ckpt_files:
        print(f"[SKIP] No .pt checkpoints found in {ckpt_dir}")
        return

    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  {experiment_name}  ({len(ckpt_files)} checkpoints, algo={cls_name})")
    print(f"{'='*60}")

    for ckpt_file in ckpt_files:
        ckpt_path = os.path.join(ckpt_dir, ckpt_file)
        try:
            algorithm, epoch = load_algorithm(cfg, algo_cls, ckpt_path, device)
        except Exception as e:
            print(f"  [ERROR] Could not load {ckpt_file}: {e}")
            continue

        for nfe in nfe_values:
            out_path = os.path.join(out_dir, f"epoch{epoch:03d}_nfe{nfe}.png")
            if os.path.exists(out_path):
                print(f"  [SKIP]  {ckpt_file} nfe={nfe:3d} (already exists)")
                continue
            try:
                images = generate_grid(algorithm, nfe, n_samples, device, SEED)
                save_image(images, out_path, nrow=8, normalize=True, value_range=(-1, 1))
                print(f"  [OK]    epoch={epoch:03d} nfe={nfe:3d} -> {out_path}")
            except Exception as e:
                print(f"  [ERROR] epoch={epoch} nfe={nfe}: {e}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate checkpoint sample grids for all (or selected) experiments."
    )
    parser.add_argument(
        "--experiments", type=str, default=None,
        help="Comma-separated experiment names (default: auto-discover all under results/)."
    )
    parser.add_argument(
        "--nfe", type=str, default="1,5,20",
        help="Comma-separated NFE values to sample at (default: 1,5,20)."
    )
    parser.add_argument(
        "--n-samples", type=int, default=64,
        help="Number of images per grid (default: 64, displayed as 8×8)."
    )
    parser.add_argument(
        "--results-dir", type=str, default=RESULTS_ROOT,
        help=f"Root of experiment results (default: {RESULTS_ROOT})."
    )
    parser.add_argument(
        "--out-dir", type=str, default=OUT_ROOT,
        help=f"Output root for sample grids (default: {OUT_ROOT})."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    nfe_values = [int(x) for x in args.nfe.split(",")]

    if args.experiments:
        experiments = [e.strip() for e in args.experiments.split(",")]
    else:
        experiments = discover_runs(args.results_dir)
        if not experiments:
            print(f"No valid experiment runs found under '{args.results_dir}'.")
            print("Run training first, or pass --experiments <name1,name2>.")
            return
        print(f"Auto-discovered {len(experiments)} run(s): {', '.join(experiments)}")

    print(f"Device     : {DEVICE}")
    print(f"NFE values : {nfe_values}")
    print(f"Samples    : {args.n_samples} (8×8 grid)")
    print(f"Output     : {args.out_dir}/")

    for exp in experiments:
        process_run(exp, args.results_dir, args.out_dir, nfe_values, args.n_samples, DEVICE)

    print(f"\nDone. Grids saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
