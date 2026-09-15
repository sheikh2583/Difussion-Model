"""
training — generic training engine.

This package contains the Trainer class, which implements the full
epoch/batch loop, optimizer and LR-scheduler management, AMP (automatic
mixed precision), checkpointing, structured logging, and evaluation
hooks.

The Trainer is completely algorithm-agnostic: it interacts with the
active algorithm solely through BaseAlgorithm.training_step(batch),
which returns a dict containing at least the key "loss".  All
algorithm-specific mathematics (noise sampling, interpolation,
Jacobian-vector products, etc.) are encapsulated inside the algorithm
implementations and are invisible to the Trainer.

Helper functions
----------------
build_optimizer(modules, cfg)  — constructs an AdamW / Adam / SGD
    optimizer over the parameters of the supplied module list.
build_scheduler(optimizer, cfg) — optionally wraps the optimizer in a
    CosineAnnealingLR or StepLR scheduler.
"""

from training.trainer import Trainer, build_optimizer, build_scheduler

__all__ = [
    "Trainer",
    "build_optimizer",
    "build_scheduler",
]
