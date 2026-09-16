"""Compatibility helpers for current and legacy project checkpoints."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def extract_model_state(checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the backbone state dict from any project checkpoint schema."""
    if "model_state" in checkpoint:
        return checkpoint["model_state"]
    if "module_state_dicts" in checkpoint:
        states = checkpoint["module_state_dicts"]
        if not states:
            raise ValueError("Checkpoint contains an empty module_state_dicts list.")
        return states[0]
    return checkpoint


def load_algorithm_state(algorithm: Any, checkpoint: Mapping[str, Any]) -> None:
    """Load backbone and extra trainable modules from current or legacy payloads."""
    modules = list(algorithm.trainable_modules())
    if not modules:
        raise ValueError("Algorithm exposes no trainable modules.")

    if "module_state_dicts" in checkpoint:
        states = checkpoint["module_state_dicts"]
        if len(states) != len(modules):
            raise ValueError(
                f"Checkpoint has {len(states)} module states, "
                f"but {algorithm.name()} expects {len(modules)}."
            )
        for module, state in zip(modules, states):
            module.load_state_dict(state)
        return

    modules[0].load_state_dict(extract_model_state(checkpoint))
    for index, module in enumerate(modules[1:]):
        key = f"extra_module_{index}_state"
        if key in checkpoint:
            module.load_state_dict(checkpoint[key])
