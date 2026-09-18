"""Compatibility helpers for current and legacy project checkpoints."""

from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any
from pathlib import Path

from utils.checkpoint_runs import find_checkpoint


def _load_module_state(module: Any, state: Mapping[str, Any]) -> None:
    """Load a module, adapting the legacy bare-Sequential key layout."""
    try:
        module.load_state_dict(state)
        return
    except RuntimeError as original_error:
        current_keys = set(module.state_dict())
        prefixed = {f"net.{key}": value for key, value in state.items()}
        if set(prefixed) == current_keys:
            module.load_state_dict(prefixed)
            return
        unprefixed = {
            key.removeprefix("net."): value for key, value in state.items()
        }
        if set(unprefixed) == current_keys:
            module.load_state_dict(unprefixed)
            return
        raise original_error


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
            _load_module_state(module, state)
    else:
        _load_module_state(modules[0], extract_model_state(checkpoint))
        for index, module in enumerate(modules[1:]):
            key = f"extra_module_{index}_state"
            if key in checkpoint:
                _load_module_state(module, checkpoint[key])

    algorithm.load_checkpoint_state(checkpoint.get("algorithm_state"))


def resolve_checkpoint_reference(path: str | Path) -> Path:
    """Resolve a legacy flat checkpoint reference to the latest numbered run.

    Config files keep their stable logical path, while new physical files live
    under ``checkpoints/run_N``. Existing paths always win.
    """
    requested = Path(path)
    if requested.is_file() or requested.parent.name != "checkpoints":
        return requested
    match = re.fullmatch(r"(.+)_epoch(\d+)\.pt", requested.name)
    if match:
        resolved = find_checkpoint(
            requested.parent.parent,
            class_name=match.group(1),
            epoch=int(match.group(2)),
        )
        if resolved is not None:
            return resolved
    return requested
