"""
CLI entry point: evaluate a checkpoint without retraining.

Usage
-----
    python evaluate.py --algorithm fm --checkpoint results/fm_cifar10/checkpoints/run_1/FlowMatchingAlgorithm_epoch100.pt --config config/fm_full.json
    python evaluate.py --algorithm mf --checkpoint results/mf_cifar10/checkpoints/run_1/MeanFlowAlgorithm_epoch100.pt --config config/mf_full.json

Add --make-plots to regenerate FID/IS vs NFE curves into results/<experiment>/metrics/plots/.
"""
import argparse
import os
import re
from pathlib import Path

import torch

from algorithms import ALGORITHM_REGISTRY
from config.config import ExperimentConfig
from data.dataset_registry import get_dataloaders_for_config
from evaluation.evaluator import Evaluator, ensure_fid_reference
from models.backbone import build_backbone
from sampling.sampler import Sampler
from utils.checkpoints import load_algorithm_state, resolve_checkpoint_reference
from utils.checkpoint_runs import run_directory_from_checkpoint
from utils.device import resolve_device
from utils.gpu_lock import acquire_gpu_lock


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a saved checkpoint.")
    parser.add_argument("--algorithm", choices=list(ALGORITHM_REGISTRY.keys()), required=True)
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to .pt checkpoint file.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--make-plots", action="store_true",
                        help="Generate metric plots after evaluation.")
    parser.add_argument("--experiment-name", type=str, default=None)
    return parser.parse_args()


def _main():
    args = parse_args()
    cfg  = ExperimentConfig.load(args.config) if args.config else ExperimentConfig()
    if args.experiment_name:
        cfg.experiment_name = args.experiment_name

    device = resolve_device(cfg)
    checkpoint_path = resolve_checkpoint_reference(args.checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint_run_dir = run_directory_from_checkpoint(checkpoint_path)
    if checkpoint_run_dir is not None:
        run_dir = str(checkpoint_run_dir)
    else:
        suffix = f"_{cfg.dataset.name}"
        run_name = (
            cfg.experiment_name
            if cfg.experiment_name.endswith(suffix)
            else f"{cfg.experiment_name}{suffix}"
        )
        run_dir = os.path.join(cfg.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    _, test_loader = get_dataloaders_for_config(cfg)
    ensure_fid_reference(cfg, test_loader, device)

    model = build_backbone(cfg.backbone, image_size=cfg.dataset.image_size)
    algorithm_cls = ALGORITHM_REGISTRY[args.algorithm]
    algorithm = algorithm_cls(model, algorithm_kwargs=cfg.algorithm_kwargs)

    # Load checkpoint — handles both raw state-dicts and trainer payloads
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    load_algorithm_state(algorithm, state)

    for m in algorithm.trainable_modules():
        m.to(device)

    run_name = os.path.basename(os.path.normpath(run_dir))
    sampler   = Sampler(algorithm, device, run_dir, run_name, cfg.seed)
    evaluator = Evaluator(cfg, run_dir, device)
    epoch_match = re.search(r"epoch(\d+)", str(checkpoint_path))
    inferred_epoch = int(epoch_match.group(1)) if epoch_match else None
    evaluator.evaluate(sampler, nfe_values=cfg.evaluation.nfe_values,
                       make_plots=args.make_plots,
                       checkpoint_path=str(checkpoint_path),
                       current_epoch=inferred_epoch)


def main():
    with acquire_gpu_lock(command="evaluate.py"):
        _main()


if __name__ == "__main__":
    main()
