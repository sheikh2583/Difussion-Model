"""
CLI entry point: build a config, pick an algorithm, run the full
train -> checkpoint -> sample -> evaluate pipeline via ExperimentRunner.

Usage:
    python train.py --algorithm mock
    python train.py --algorithm fm --config path/to/config.json
    python train.py --algorithm mf --config path/to/config.json
"""
import argparse

from algorithms.flow_matching import FlowMatchingAlgorithm
from algorithms.flow_matching_lognorm import FlowMatchingLognormAlgorithm
from algorithms.mean_flow import MeanFlowAlgorithm
from algorithms.mock import MockAlgorithm
from config.config import ExperimentConfig
from experiments.runner import ExperimentRunner

ALGORITHM_REGISTRY = {
    "mock": MockAlgorithm,
    "fm": FlowMatchingAlgorithm,
    "fm_lognorm": FlowMatchingLognormAlgorithm,
    "mf": MeanFlowAlgorithm,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train an image-generation algorithm.")
    parser.add_argument("--algorithm", choices=list(ALGORITHM_REGISTRY.keys()), required=True)
    parser.add_argument("--config", type=str, default=None,
                         help="Path to a JSON config file. If omitted, uses defaults.")
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = ExperimentConfig.load(args.config) if args.config else ExperimentConfig()
    if args.experiment_name:
        cfg.experiment_name = args.experiment_name
    if args.epochs:
        cfg.epochs = args.epochs

    algorithm_cls = ALGORITHM_REGISTRY[args.algorithm]
    runner = ExperimentRunner(cfg, algorithm_cls)

    cfg.save(f"{runner.run_dir}/config.json")
    runner.run_full()


if __name__ == "__main__":
    main()
