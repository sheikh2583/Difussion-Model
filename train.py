"""
CLI entry point: train any registered algorithm.

Usage
-----
    python train.py --algorithm fm
    python train.py --algorithm fm_lognorm --config config/fm_lognorm_full.json
    python train.py --algorithm mf         --config config/mf_full.json
    python train.py --algorithm mf_distill --config config/mf_distill_full.json

All four algorithms share the same pipeline (ExperimentRunner → Trainer →
Sampler → Evaluator).  Algorithm selection is the only thing that differs.
"""
import argparse

from algorithms import ALGORITHM_REGISTRY
from config.config import ExperimentConfig
from experiments.runner import ExperimentRunner


def parse_args():
    parser = argparse.ArgumentParser(description="Train a generative-model algorithm on CIFAR-10.")
    parser.add_argument(
        "--algorithm",
        choices=list(ALGORITHM_REGISTRY.keys()),
        required=True,
        help="Algorithm to train: fm | fm_lognorm | mf | mf_distill",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to a JSON config preset (e.g. config/mf_full.json). "
             "If omitted, uses ExperimentConfig defaults.",
    )
    parser.add_argument("--experiment-name", type=str, default=None,
                        help="Override experiment_name from config.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epochs from config.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg  = ExperimentConfig.load(args.config) if args.config else ExperimentConfig()
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
