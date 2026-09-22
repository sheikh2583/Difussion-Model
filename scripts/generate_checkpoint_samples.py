"""
generate_checkpoint_samples.py
-------------------------------
Generate image grids from every saved checkpoint for one or more
experiment runs.  Works with any registered algorithm; no
hardware-specific settings.

Usage (run from project root, venv activated):
    # Auto-discover all runs under results/:
    python scripts/generate_checkpoint_samples.py

    # Auto-discover only CelebA pixel and latent runs:
    python scripts/generate_checkpoint_samples.py --dataset-family celeba

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
import json
import os
import re
import sys
from pathlib import Path

# Allow running from scripts/ subdirectory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torchvision.utils import save_image

from algorithms import ALGORITHM_REGISTRY
from config.config import ExperimentConfig
from models.backbone import build_backbone
from utils.checkpoint_runs import latest_checkpoint_run_directory
from utils.checkpoints import load_algorithm_state
from utils.gpu_lock import DEFAULT_LOCK_PATH, acquire_gpu_lock

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


def _run_dataset(run_dir: str) -> str | None:
    """Read the dataset identity without constructing a training config."""
    try:
        payload = json.loads(Path(run_dir, "config.json").read_text(encoding="utf-8"))
        dataset = payload.get("dataset", {})
        if isinstance(dataset, dict) and dataset.get("name"):
            return str(dataset["name"]).lower()
    except (OSError, ValueError, TypeError):
        return None
    return None


def discover_runs(results_root: str, dataset_family: str = "all"):
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
            dataset = _run_dataset(run_dir)
            matches_family = (
                dataset_family == "all"
                or dataset_family == dataset
                or (dataset_family == "celeba" and dataset == "celeba_latent")
            )
            if matches_family:
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
    """Return sampled model states; latent runs still require codec decoding."""
    torch.manual_seed(seed)
    with torch.no_grad():
        images = algorithm.sample(n_samples, nfe, device)
    return images


def load_latent_decoder(cfg, device):
    """Load the configured frozen codec only for latent-space experiments."""
    if cfg.dataset.name != "celeba_latent":
        return None
    if not cfg.dataset.codec_checkpoint:
        raise ValueError("Latent checkpoint sampling requires dataset.codec_checkpoint")
    from codec.codec_factory import load_codec

    return load_codec(cfg.dataset.codec_checkpoint, device, require_frozen=True)


@torch.no_grad()
def decode_for_display(samples, codec, batch_size=32):
    """Decode normalized latent states in bounded batches before image saving."""
    if codec is None:
        images = samples
    else:
        images = torch.cat(
            [
                codec.decode_normalised(samples[start : start + batch_size]).cpu()
                for start in range(0, samples.shape[0], batch_size)
            ],
            dim=0,
        )
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(
            "Checkpoint samples must decode to an RGB tensor before save_image; "
            f"got shape {tuple(images.shape)}"
        )
    return images.cpu()


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
    try:
        codec = load_latent_decoder(cfg, device)
    except Exception as error:
        print(f"[SKIP] Could not load latent decoder for {experiment_name}: {error}")
        return
    def checkpoint_epoch(filename):
        match = re.search(r"_epoch(\d+)\.pt$", filename)
        return int(match.group(1)) if match else -1

    ckpt_files = sorted(
        (
            f
            for f in os.listdir(ckpt_dir)
            if f.startswith(cls_name) and f.endswith(".pt")
        ),
        key=checkpoint_epoch,
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
                samples = generate_grid(algorithm, nfe, n_samples, device, SEED)
                images = decode_for_display(samples, codec)
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
        "--dataset-family", choices=("all", "cifar10", "celeba"), default="all",
        help=(
            "Filter auto-discovery by dataset. 'celeba' includes separate pixel "
            "and celeba_latent runs (default: all)."
        ),
    )
    parser.add_argument(
        "--out-dir", type=str, default=OUT_ROOT,
        help=f"Output root for sample grids (default: {OUT_ROOT})."
    )
    parser.add_argument(
        "--lock-file", type=Path, default=DEFAULT_LOCK_PATH,
        help=f"Shared GPU workflow lock (default: {DEFAULT_LOCK_PATH}).",
    )
    return parser.parse_args()


def main(args=None):
    args = args or parse_args()
    nfe_values = [int(x) for x in args.nfe.split(",")]

    if args.experiments:
        experiments = [e.strip() for e in args.experiments.split(",")]
    else:
        experiments = discover_runs(args.results_dir, args.dataset_family)
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
    cli_args = parse_args()
    with acquire_gpu_lock(cli_args.lock_file, command="generate_checkpoint_samples.py"):
        main(cli_args)
