"""
Single source of truth for all experimental configuration.

Design rule (enforced by convention, see experiments/runner.py for the
runtime assertion): `algorithm_kwargs` may only contain parameters that
are inherently required by a specific algorithm (e.g. a distillation
coefficient). It must NEVER be used to override any field already
present in this shared config (batch size, epochs, optimizer, lr,
scheduler, AMP, backbone, dataset, seed, evaluation settings, etc).
"""
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
import json
import os


@dataclass
class BackboneConfig:
    # ``legacy_cifar_unet`` freezes the checkpoint-era CIFAR implementation;
    # ``simple_unet`` is the normal/current implementation.
    name: str = "simple_unet"
    in_channels: int = 3
    base_channels: int = 64
    channel_mults: List[int] = field(default_factory=lambda: [1, 2, 2])
    num_res_blocks: int = 2
    time_embed_dim: int = 256
    # Pixel-space runs preserve the historical [-1, 1] sample clamp. Latent
    # runs disable it because normalized codec latents are intentionally
    # unbounded.
    sample_clamp: bool = True


@dataclass
class DatasetConfig:
    name: str = "cifar10"
    root: str = "./data/raw"
    image_size: int = 32
    num_workers: int = 4
    # Optional fields used by content-addressed latent datasets. Empty values
    # keep every existing pixel configuration unchanged.
    cache_dir: str = ""
    split: str = ""
    codec_checkpoint: str = ""


@dataclass
class OptimConfig:
    optimizer: str = "adamw"
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    # Optional gradient clipping applied after AMP unscale and before
    # optimizer.step().  None (the default) disables clipping so that
    # all existing configs are unaffected.
    gradient_clip_norm: Optional[float] = None
    scheduler: str = "none"  # "none" | "cosine" | "step"
    scheduler_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalConfig:
    metrics: List[str] = field(default_factory=lambda: ["fid", "is"])
    num_generated_samples: int = 5000
    nfe_values: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 100])
    eval_frequency_epochs: int = 10
    fid_reference_cache: str = "./results/metrics/fid_reference_stats.npz"


@dataclass
class ExperimentConfig:
    experiment_name: str = "unnamed_experiment"
    output_dir: str = "./results"
    seed: int = 0
    device: str = "cuda"
    amp: bool = True

    batch_size: int = 128
    epochs: int = 100
    checkpoint_frequency_epochs: int = 10

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)

    # ONLY algorithm-inherent parameters go here (see module docstring).
    # Shared experimental controls above must never be duplicated or
    # shadowed inside this dict.
    algorithm_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Fields that algorithm_kwargs is forbidden from containing.
    # Used by experiments/runner.py to assert fairness at run time.
    _protected_keys: List[str] = field(default_factory=lambda: [
        "batch_size", "epochs", "checkpoint_frequency_epochs",
        "seed", "device", "amp", "dataset", "backbone", "optim",
        "evaluation", "output_dir",
    ])

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @staticmethod
    def load(path: str) -> "ExperimentConfig":
        with open(path, "r") as f:
            raw = json.load(f)
        return ExperimentConfig(
            experiment_name=raw.get("experiment_name", "unnamed_experiment"),
            output_dir=raw.get("output_dir", "./results"),
            seed=raw.get("seed", 0),
            device=raw.get("device", "cuda"),
            amp=raw.get("amp", True),
            batch_size=raw.get("batch_size", 128),
            epochs=raw.get("epochs", 100),
            checkpoint_frequency_epochs=raw.get("checkpoint_frequency_epochs", 10),
            dataset=DatasetConfig(**raw.get("dataset", {})),
            backbone=BackboneConfig(**raw.get("backbone", {})),
            optim=OptimConfig(**raw.get("optim", {})),
            evaluation=EvalConfig(**raw.get("evaluation", {})),
            algorithm_kwargs=raw.get("algorithm_kwargs", {}),
        )
