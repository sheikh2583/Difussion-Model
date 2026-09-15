"""
config — single source of truth for all experimental configuration.

All hyperparameters that govern a run (dataset, backbone, optimizer,
scheduler, AMP, evaluation settings, seed, etc.) live exclusively in
ExperimentConfig.  Algorithms may declare their own inherent parameters
in ExperimentConfig.algorithm_kwargs, but that dict must never shadow
any field already present in the shared config — this invariant is
enforced at runtime by experiments/runner.py.

Dataclasses
-----------
ExperimentConfig   — top-level config, serialisable to / from JSON.
BackboneConfig     — architecture hyperparameters for the shared UNet.
DatasetConfig      — dataset root, image size, worker count.
OptimConfig        — optimizer type, learning rate, weight decay,
                     scheduler name and its keyword arguments.
EvalConfig         — list of metrics, sample count, NFE values,
                     evaluation frequency, FID reference cache path.

Typical usage
-------------
    # Load from JSON (e.g. one of the presets in config/):
    cfg = ExperimentConfig.load("config/fm_full.json")

    # Construct with defaults then override:
    cfg = ExperimentConfig(epochs=50, seed=42)

    # Persist alongside experiment outputs:
    cfg.save(f"{run_dir}/config.json")
"""

from config.config import (
    ExperimentConfig,
    BackboneConfig,
    DatasetConfig,
    OptimConfig,
    EvalConfig,
)

__all__ = [
    "ExperimentConfig",
    "BackboneConfig",
    "DatasetConfig",
    "OptimConfig",
    "EvalConfig",
]
