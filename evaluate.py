"""
CLI entry point: evaluate a checkpoint without retraining.

Usage
-----
    python evaluate.py --algorithm fm         --checkpoint results/fm_cifar10/checkpoints/FlowMatchingAlgorithm_epoch100.pt --config config/fm_full.json
    python evaluate.py --algorithm fm_lognorm  --checkpoint results/fm_lognorm_cifar10/checkpoints/FlowMatchingLognormAlgorithm_epoch100.pt --config config/fm_lognorm_full.json
    python evaluate.py --algorithm mf           --checkpoint results/mf_cifar10/checkpoints/MeanFlowAlgorithm_epoch100.pt --config config/mf_full.json
    python evaluate.py --algorithm mf_distill   --checkpoint results/mf_distill_cifar10/checkpoints/MeanFlowDistillAlgorithm_epoch100.pt --config config/mf_distill_full.json
    python evaluate.py --algorithm consistency  --checkpoint results/consistency_cifar10/checkpoints/ConsistencyAlgorithm_epoch100.pt --config config/consistency_full.json
    python evaluate.py --algorithm reflow       --checkpoint results/reflow_cifar10/checkpoints/ReflowAlgorithm_epoch100.pt --config config/reflow_full.json

Add --make-plots to regenerate FID/IS vs NFE curves into results/<experiment>/metrics/plots/.
"""
import argparse
import os

import torch

from algorithms import ALGORITHM_REGISTRY
from config.config import ExperimentConfig
from data.dataset_registry import get_dataloaders_for_config
from evaluation.evaluator import Evaluator, ensure_fid_reference
from models.backbone import build_backbone
from sampling.sampler import Sampler
from utils.device import resolve_device


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


def main():
    args = parse_args()
    cfg  = ExperimentConfig.load(args.config) if args.config else ExperimentConfig()
    if args.experiment_name:
        cfg.experiment_name = args.experiment_name

    device = resolve_device(cfg)
    run_dir = os.path.join(cfg.output_dir, cfg.experiment_name)
    os.makedirs(run_dir, exist_ok=True)

    _, test_loader = get_dataloaders_for_config(cfg)
    ensure_fid_reference(cfg, test_loader, device)

    model = build_backbone(cfg.backbone, image_size=cfg.dataset.image_size)
    algorithm_cls = ALGORITHM_REGISTRY[args.algorithm]
    algorithm = algorithm_cls(model, algorithm_kwargs=cfg.algorithm_kwargs)

    # Load checkpoint — handles both raw state-dicts and trainer payloads
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state.get("model_state", state))
    # Restore extra modules (e.g. r_embed) if present in checkpoint
    extra_modules = algorithm.trainable_modules()[1:]
    for i, m in enumerate(extra_modules):
        key = f"extra_module_{i}_state"
        if key in state:
            m.load_state_dict(state[key])

    for m in algorithm.trainable_modules():
        m.to(device)

    sampler   = Sampler(algorithm, device, run_dir, cfg.experiment_name, cfg.seed)
    evaluator = Evaluator(cfg, run_dir, device)
    evaluator.evaluate(sampler, nfe_values=cfg.evaluation.nfe_values,
                       make_plots=args.make_plots)


if __name__ == "__main__":
    main()
