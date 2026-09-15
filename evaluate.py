"""
CLI entry point: load a checkpoint and run sampling + evaluation only
(no training). Useful for re-evaluating at different NFE values or
regenerating plots without retraining.

Usage:
    python evaluate.py --algorithm mock --checkpoint results/.../checkpoints/MockAlgorithm_epoch10.pt --config results/.../config.json
"""
import argparse

import torch

from algorithms.flow_matching import FlowMatchingAlgorithm
from algorithms.flow_matching_lognorm import FlowMatchingLognormAlgorithm
from algorithms.mean_flow import MeanFlowAlgorithm
from algorithms.mock import MockAlgorithm
from config.config import ExperimentConfig
from experiments.runner import ExperimentRunner
from utils.plots import (
    plot_loss_vs_epoch, plot_training_time_comparison, plot_sampling_time_vs_nfe,
    plot_fid_vs_nfe, plot_is_vs_nfe, plot_gpu_memory_comparison, plot_fid_vs_sampling_time,
)

ALGORITHM_REGISTRY = {
    "mock": MockAlgorithm,
    "fm": FlowMatchingAlgorithm,
    "fm_lognorm": FlowMatchingLognormAlgorithm,
    "mf": MeanFlowAlgorithm,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint.")
    parser.add_argument("--algorithm", choices=list(ALGORITHM_REGISTRY.keys()), required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--make-plots", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = ExperimentConfig.load(args.config)

    algorithm_cls = ALGORITHM_REGISTRY[args.algorithm]
    runner = ExperimentRunner(cfg, algorithm_cls)

    # PyTorch >=2.6 defaults torch.load to weights_only=True, which refuses
    # to unpickle the custom ExperimentConfig object stored in our checkpoints.
    # Safe to disable here since we only ever load checkpoints this project
    # itself wrote.
    ckpt = torch.load(args.checkpoint, map_location=runner.device, weights_only=False)
    for module, state_dict in zip(runner.algorithm.trainable_modules(), ckpt["module_state_dicts"]):
        module.to(runner.device)
        module.load_state_dict(state_dict)

    runner.sample()
    runner.evaluate()

    if args.make_plots:
        jsonl_path = f"{runner.run_dir}/metrics/{cfg.experiment_name}.jsonl"
        plots_dir = f"{runner.run_dir}/metrics/plots"
        plot_loss_vs_epoch(jsonl_path, f"{plots_dir}/loss_vs_epoch.png")
        plot_training_time_comparison(jsonl_path, f"{plots_dir}/training_time.png")
        plot_sampling_time_vs_nfe(jsonl_path, f"{plots_dir}/sampling_time_vs_nfe.png")
        plot_fid_vs_nfe(jsonl_path, f"{plots_dir}/fid_vs_nfe.png")
        plot_is_vs_nfe(jsonl_path, f"{plots_dir}/is_vs_nfe.png")
        plot_gpu_memory_comparison(jsonl_path, f"{plots_dir}/gpu_memory.png")
        plot_fid_vs_sampling_time(jsonl_path, f"{plots_dir}/fid_vs_sampling_time.png")


if __name__ == "__main__":
    main()
