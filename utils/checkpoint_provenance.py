"""Build and validate portable checkpoint identity metadata.

Only fields that must remain identical across a resume are included. Runtime
controls such as the target epoch, worker count, dataset root, and evaluation
settings are intentionally excluded so checkpoints can move between machines
and a completed run can be extended safely.
"""

from __future__ import annotations

import hashlib
import json
import os
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Optional, Type

import torch

from algorithms.base import BaseAlgorithm
from config.config import ExperimentConfig


PROVENANCE_VERSION = 1


class ResumeCompatibilityError(ValueError):
    """Raised before weights are loaded from an incompatible checkpoint."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def build_provenance(
    cfg: ExperimentConfig,
    algorithm_cls: Type[BaseAlgorithm],
    algorithm_key: Optional[str] = None,
) -> dict[str, Any]:
    """Return versioned, JSON-safe resume identity for a new checkpoint."""
    identity = {
        "algorithm_key": algorithm_key,
        "algorithm_class": f"{algorithm_cls.__module__}.{algorithm_cls.__qualname__}",
        "run_variant": cfg.experiment_name,
        "dataset": {
            "name": cfg.dataset.name,
            "image_size": cfg.dataset.image_size,
        },
        "backbone": asdict(cfg.backbone),
        "algorithm_kwargs": cfg.algorithm_kwargs,
    }
    digest = hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()
    provenance = {
        "version": PROVENANCE_VERSION,
        "identity": identity,
        "identity_sha256": digest,
    }
    source_identity = os.environ.get("DIFFUSION_SOURCE_IDENTITY")
    if source_identity:
        provenance["source_identity_sha256"] = source_identity
    return provenance


def _differences(expected: Any, actual: Any, prefix: str = "") -> list[str]:
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        differences: list[str] = []
        for key in sorted(set(expected) | set(actual)):
            field = f"{prefix}.{key}" if prefix else str(key)
            if key not in expected:
                differences.append(f"{field}: unexpected value {actual[key]!r}")
            elif key not in actual:
                differences.append(f"{field}: missing (expected {expected[key]!r})")
            else:
                differences.extend(_differences(expected[key], actual[key], field))
        return differences
    if expected != actual:
        return [f"{prefix}: expected {expected!r}, found {actual!r}"]
    return []


def validate_provenance(
    actual: Optional[Mapping[str, Any]],
    expected: Mapping[str, Any],
    checkpoint_path: Path,
) -> bool:
    """Validate metadata, returning False with a warning for legacy payloads."""
    if actual is None:
        warnings.warn(
            f"Legacy checkpoint has no provenance metadata: {checkpoint_path}. "
            "Compatibility cannot be proven; loading remains enabled for old runs.",
            UserWarning,
            stacklevel=2,
        )
        return False

    if actual.get("version") != PROVENANCE_VERSION:
        raise ResumeCompatibilityError(
            f"Cannot resume {checkpoint_path}: unsupported provenance version "
            f"{actual.get('version')!r}; expected {PROVENANCE_VERSION}."
        )

    expected_identity = expected["identity"]
    actual_identity = actual.get("identity")
    differences = _differences(expected_identity, actual_identity, "identity")
    expected_digest = hashlib.sha256(
        _canonical_json(expected_identity).encode("utf-8")
    ).hexdigest()
    if actual.get("identity_sha256") != expected_digest:
        differences.append("identity_sha256: metadata digest is missing or invalid")
    if differences:
        details = "\n  - ".join(differences)
        raise ResumeCompatibilityError(
            f"Cannot resume incompatible checkpoint {checkpoint_path}:\n  - {details}"
        )
    expected_source = expected.get("source_identity_sha256")
    actual_source = actual.get("source_identity_sha256")
    if expected_source and actual_source and actual_source != expected_source:
        raise ResumeCompatibilityError(
            f"Cannot resume {checkpoint_path}: checkpoint source identity "
            f"{actual_source} differs from current suite identity {expected_source}."
        )
    return True


def validate_checkpoint_file(
    checkpoint_path: Path,
    expected: Mapping[str, Any],
) -> bool:
    """Read only checkpoint metadata and reject incompatible resumes early."""
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        return validate_provenance(None, expected, checkpoint_path)
    return validate_provenance(payload.get("provenance"), expected, checkpoint_path)
