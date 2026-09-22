#!/usr/bin/env python3
"""Add non-destructive pixel/latent metadata to existing central logs.

This is the safe bridge while a training suite is active: it writes or merges
``<log>.meta.json`` sidecars but never renames or edits a transcript. Physical
log migration is deliberately deferred until all writers have exited.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALGORITHMS = (
    "fm_lognorm", "mf_distill", "consistency", "reflow", "fm", "mf"
)


def classify_name(path: Path) -> dict[str, str]:
    """Infer only unambiguous diffusion-log metadata from a filename."""
    stem = path.name.removesuffix(".log")
    path_text = path.as_posix().lower()
    if "/scratch_vae/" in path_text or (
        "/codec/" in path_text and "scratch_kl_vae" in stem
    ):
        return {
            "training_type": "scratch_codec",
            "representation_space": "codec",
            "dataset": "celeba",
            "algorithm": "scratch_kl_vae",
        }
    if "/codecs/" in path_text or (
        "/codec/" in path_text and "pretrained_vq_f4" in stem
    ):
        return {
            "training_type": "codec_validation",
            "representation_space": "codec",
            "dataset": "celeba",
            "algorithm": "pretrained_vq_f4",
        }
    if stem.startswith("tournament_run_"):
        return {
            "training_type": "suite_orchestration",
            # Historical tournaments covered CIFAR-10 and CelebA pixel runs.
            "representation_space": "pixel",
            "dataset": "multiple",
            "algorithm": "multiple",
        }
    if stem.startswith("celeba_latent_"):
        dataset, space, remainder = "celeba_latent", "latent", stem[14:]
        training_type = "latent_diffusion"
    elif stem.startswith("cifar10_pixel_"):
        dataset, space, remainder = "cifar10", "pixel", stem[14:]
        training_type = "pixel_diffusion"
    elif stem.startswith("celeba_pixel_"):
        dataset, space, remainder = "celeba", "pixel", stem[13:]
        training_type = "pixel_diffusion"
    elif stem.startswith("cifar10_"):
        dataset, space, remainder = "cifar10", "pixel", stem[8:]
        training_type = "pixel_diffusion"
    elif stem.startswith("celeba_"):
        dataset, space, remainder = "celeba", "pixel", stem[7:]
        training_type = "pixel_diffusion"
    elif "celeba_latent" in path_text:
        dataset, space, remainder = "celeba_latent", "latent", path_text
        training_type = "latent_diffusion"
    elif "cifar10" in path_text:
        dataset, space, remainder = "cifar10", "pixel", path_text
        training_type = "pixel_diffusion"
    elif "celeba" in path_text:
        dataset, space, remainder = "celeba", "pixel", path_text
        training_type = "pixel_diffusion"
    else:
        return {}

    algorithm = next((candidate for candidate in ALGORITHMS if (
        remainder.startswith(candidate + "_")
        or f"/{candidate}_" in remainder
        or f".{candidate}" in remainder
    )), "")
    result = {
        "training_type": training_type,
        "representation_space": space,
        "dataset": dataset,
    }
    if algorithm:
        result["algorithm"] = algorithm
    elif stem == "cifar10_training":
        result["algorithm"] = "multiple"
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.",
        suffix=".tmp", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def annotate(
    log_path: Path,
    project_root: Path,
    apply: bool,
    identification: dict[str, Any] | None = None,
) -> bool:
    inferred = classify_name(log_path)
    if not inferred:
        return False
    sidecar = log_path.with_suffix(log_path.suffix + ".meta.json")
    existing: dict[str, Any] = {}
    if sidecar.is_file():
        try:
            loaded = json.loads(sidecar.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing = loaded
        except (OSError, json.JSONDecodeError):
            raise ValueError(f"Refusing to replace invalid sidecar: {sidecar}")
    conflicts = {
        key: (existing[key], value)
        for key, value in inferred.items()
        if key in existing and existing[key] != value
    }
    if conflicts:
        raise ValueError(f"Metadata conflict for {log_path}: {conflicts}")

    payload = {
        **existing,
        **inferred,
        **(identification or {}),
        "log_path_at_annotation": log_path.relative_to(project_root).as_posix(),
        "annotation_schema_version": 2,
        "annotation_source": "scripts/annotate_training_log_spaces.py",
        "transcript_bytes": log_path.stat().st_size,
        "transcript_sha256": _sha256(log_path),
    }
    if "annotated_utc" not in payload:
        payload["annotated_utc"] = datetime.now(timezone.utc).isoformat()
    if apply:
        _atomic_json(sidecar, payload)
    print(
        f"[{'WRITE' if apply else 'PLAN'}] {inferred['representation_space']}: "
        f"{log_path.relative_to(project_root)}"
    )
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--apply", action="store_true",
        help="Write sidecars. Without this flag the command is a dry-run.",
    )
    parser.add_argument("--machine-label")
    parser.add_argument("--gpu-name")
    parser.add_argument("--gpu-memory-gb", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    roots = (project_root / "training_logs", project_root / "results")
    logs = sorted(
        path for root in roots if root.is_dir() for path in root.rglob("*.log")
    )
    supplied = (args.machine_label, args.gpu_name, args.gpu_memory_gb)
    if any(value is not None for value in supplied) and not all(
        value is not None for value in supplied
    ):
        raise SystemExit(
            "--machine-label, --gpu-name, and --gpu-memory-gb must be supplied together"
        )
    identification = None
    if all(value is not None for value in supplied):
        identification = {
            "machine_label": args.machine_label,
            "gpu_name": args.gpu_name,
            "gpu_memory_gb": args.gpu_memory_gb,
            "identification_source": "operator-confirmed historical lab hardware",
        }
    classified = sum(
        annotate(path, project_root, args.apply, identification) for path in logs
    )
    print(f"Classified {classified} of {len(logs)} discovered training logs.")
    if not args.apply:
        print("Dry-run only; pass --apply to write sidecars.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
