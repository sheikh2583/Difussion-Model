"""
experiments — end-to-end experiment orchestration.

This package contains ExperimentRunner, the top-level coordinator that
wires every other component together (config, data, model, algorithm,
trainer, sampler, evaluator) and enforces the experimental-fairness
constraints:

  - algorithm_kwargs must not shadow any protected shared config field;
  - the backbone is built once from the shared config and passed to the
    algorithm (preventing silent architecture divergence);
  - dataloaders are built once and reused across train / sample /
    evaluate phases;
  - the FID reference statistics are cached and reused across all
    algorithm runs in the same output directory.

Typical usage (via CLI entry points train.py / evaluate.py):
    runner = ExperimentRunner(cfg, FlowMatchingAlgorithm)
    runner.run_full()   # train → sample → evaluate
"""

from experiments.runner import ExperimentRunner

__all__ = ["ExperimentRunner"]
