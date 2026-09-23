"""Generate restart-safe latent FM Reflow pairs (operator command only).

Pairs are Gaussian ``z1`` and latent FM-generated ``x0``. CelebA pixels, the
latent cache, and the codec are not inputs to this operation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from algorithms.flow_matching import FlowMatchingAlgorithm
from config.config import ExperimentConfig
from models.backbone import build_backbone
from utils.checkpoint_provenance import build_provenance, validate_provenance
from utils.checkpoints import extract_model_state
from utils.device import resolve_device
from utils.gpu_lock import DEFAULT_LOCK_PATH, acquire_gpu_lock


PAIR_SHAPE = (3, 16, 16)
SCHEMA_VERSION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate latent FM Reflow pairs")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="data/reflow_pairs_celeba_latent.pt")
    parser.add_argument("--n-pairs", type=int, default=50000)
    parser.add_argument("--nfe", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lock-file", default=str(DEFAULT_LOCK_PATH))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow-source-identity-mismatch",
        action="store_true",
        help=(
            "Accept a teacher produced by a different source suite only when its "
            "structured algorithm, dataset, and backbone identity still matches"
        ),
    )
    return parser.parse_args()


def atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_fm_config(cfg: ExperimentConfig) -> None:
    if cfg.dataset.name != "celeba_latent":
        raise ValueError(f"Expected dataset.name='celeba_latent', got {cfg.dataset.name!r}")
    if cfg.dataset.image_size != 16 or cfg.backbone.in_channels != 3:
        raise ValueError("Latent FM config must specify three channels at spatial size 16")
    if cfg.backbone.sample_clamp:
        raise ValueError("Latent FM config must set backbone.sample_clamp=false")


@torch.no_grad()
def integrate_fm(model: torch.nn.Module, z1: torch.Tensor, nfe: int) -> torch.Tensor:
    x = z1.clone()
    step = 1.0 / nfe
    for index in range(nfe):
        t = torch.full(
            (x.shape[0],), 1.0 - index * step, device=x.device, dtype=torch.float32
        )
        x = x - model(x, t) * step
    if not torch.isfinite(x).all():
        raise FloatingPointError("Latent FM generated NaN or Inf endpoints")
    return x


def _validate_chunk(payload: Mapping[str, Any], expected_count: int) -> None:
    for key in ("z1", "x0"):
        tensor = payload.get(key)
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Reflow chunk is missing tensor {key!r}")
        if tensor.dtype != torch.float32 or tuple(tensor.shape) != (expected_count, *PAIR_SHAPE):
            raise ValueError(
                f"Invalid {key} chunk shape/dtype: {tuple(tensor.shape)} {tensor.dtype}"
            )
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Reflow chunk {key!r} contains NaN or Inf")


def assemble_chunks(chunk_dir: Path, metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and assemble all restart chunks into the existing Reflow schema."""
    pair_count = int(metadata["pair_count"])
    chunk_size = int(metadata["chunk_size"])
    z1_all = torch.empty(pair_count, *PAIR_SHAPE, dtype=torch.float32)
    x0_all = torch.empty_like(z1_all)
    for start in range(0, pair_count, chunk_size):
        end = min(start + chunk_size, pair_count)
        path = chunk_dir / f"chunk_{start:08d}_{end:08d}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing Reflow generation chunk: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("start") != start or payload.get("end") != end:
            raise ValueError(f"Chunk bounds do not match filename: {path}")
        _validate_chunk(payload, end - start)
        z1_all[start:end].copy_(payload["z1"])
        x0_all[start:end].copy_(payload["x0"])
    return {"z1": z1_all, "x0": x0_all, "metadata": dict(metadata)}


def _generate(args: argparse.Namespace) -> None:
    for name in ("n_pairs", "nfe", "batch_size", "chunk_size"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 1")
    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    output_path = Path(args.output).resolve()
    if not config_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError("Both --config and --checkpoint must name existing files")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite intentionally: {output_path}")

    cfg = ExperimentConfig.load(str(config_path))
    _validate_fm_config(cfg)
    expected_provenance = build_provenance(cfg, FlowMatchingAlgorithm, "fm")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("FM checkpoint must be a mapping with model state and provenance")
    provenance_ok = validate_provenance(
        checkpoint.get("provenance"),
        expected_provenance,
        checkpoint_path,
        allow_source_identity_mismatch=args.allow_source_identity_mismatch,
    )
    if not provenance_ok:
        raise ValueError("Latent pair generation requires a provenance-bearing FM checkpoint")

    device = resolve_device(cfg)
    model = build_backbone(cfg.backbone, image_size=cfg.dataset.image_size)
    model.load_state_dict(extract_model_state(checkpoint), strict=True)
    model.to(device).eval()

    checkpoint_provenance = checkpoint["provenance"]
    checkpoint_source_identity = checkpoint_provenance.get("source_identity_sha256")
    current_source_identity = expected_provenance.get("source_identity_sha256")
    source_identity_mismatch = bool(
        checkpoint_source_identity
        and current_source_identity
        and checkpoint_source_identity != current_source_identity
    )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "fm_checkpoint_sha256": sha256_file(checkpoint_path),
        "fm_checkpoint": str(checkpoint_path),
        "nfe": args.nfe,
        "seed": args.seed,
        "shape": [args.n_pairs, *PAIR_SHAPE],
        "pair_count": args.n_pairs,
        "chunk_size": args.chunk_size,
        "config_sha256": sha256_file(config_path),
        "config_identity_sha256": expected_provenance["identity_sha256"],
        "fm_checkpoint_source_identity_sha256": checkpoint_source_identity,
        "pair_generation_source_identity_sha256": current_source_identity,
        "source_identity_mismatch_accepted": (
            source_identity_mismatch and args.allow_source_identity_mismatch
        ),
        "dataset": "celeba_latent",
    }
    chunk_dir = output_path.with_name(f"{output_path.name}.chunks")
    manifest_path = chunk_dir / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != metadata:
            raise ValueError(
                f"Restart manifest differs from this request: {manifest_path}. "
                "Use a different output path or remove it after explicit inspection."
            )
    else:
        atomic_json_save(metadata, manifest_path)

    for start in range(0, args.n_pairs, args.chunk_size):
        end = min(start + args.chunk_size, args.n_pairs)
        chunk_path = chunk_dir / f"chunk_{start:08d}_{end:08d}.pt"
        if chunk_path.exists():
            existing = torch.load(chunk_path, map_location="cpu", weights_only=True)
            _validate_chunk(existing, end - start)
            if existing.get("start") != start or existing.get("end") != end:
                raise ValueError(f"Invalid restart chunk bounds: {chunk_path}")
            print(f"verified existing chunk {start}:{end}", flush=True)
            continue

        cpu_generator = torch.Generator().manual_seed(args.seed + start)
        z_parts: list[torch.Tensor] = []
        x_parts: list[torch.Tensor] = []
        for offset in range(start, end, args.batch_size):
            count = min(args.batch_size, end - offset)
            z1 = torch.randn(count, *PAIR_SHAPE, generator=cpu_generator).to(device)
            x0 = integrate_fm(model, z1, args.nfe)
            z_parts.append(z1.float().cpu())
            x_parts.append(x0.float().cpu())
        payload = {
            "start": start, "end": end,
            "z1": torch.cat(z_parts), "x0": torch.cat(x_parts),
        }
        _validate_chunk(payload, end - start)
        atomic_torch_save(payload, chunk_path)
        print(f"generated chunk {start}:{end}", flush=True)

    assembled = assemble_chunks(chunk_dir, metadata)
    atomic_torch_save(assembled, output_path)
    atomic_json_save(metadata, output_path.with_suffix(output_path.suffix + ".json"))
    print(f"Saved {args.n_pairs} latent Reflow pairs to {output_path}")


def main() -> None:
    args = parse_args()
    with acquire_gpu_lock(
        Path(args.lock_file).resolve(), command="generate_reflow_pairs_latent.py"
    ):
        _generate(args)


if __name__ == "__main__":
    main()
