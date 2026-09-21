"""Verified, content-addressed CelebA latent-cache dataset.

The cache producer owns serialization; this reader deliberately depends only
on the shared manifest contract. A cache directory may contain multiple codec
identities and both dataset splits without name-based selection.
"""
from __future__ import annotations

import hashlib
import json
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from config.config import DatasetConfig


EXPECTED_SHAPE = (4, 16, 16)
_SPLIT_ALIASES = {"validation": "valid", "val": "valid"}


def _canonical_split(split: str) -> str:
    value = _SPLIT_ALIASES.get(split.lower(), split.lower())
    if value not in {"train", "valid"}:
        raise ValueError(f"Latent cache split must be 'train' or 'valid', got {split!r}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _nested(metadata: dict[str, Any], *paths: str) -> Any:
    for dotted in paths:
        value: Any = metadata
        for component in dotted.split("."):
            if not isinstance(value, dict) or component not in value:
                break
            value = value[component]
        else:
            return value
    return None


def _manifest_candidates(cache_dir: Path) -> Iterable[Path]:
    if cache_dir.is_file() and cache_dir.name == "manifest.json":
        yield cache_dir
        return
    if cache_dir.is_dir():
        yield from sorted(cache_dir.rglob("manifest.json"))


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid latent-cache manifest {path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"Latent-cache manifest must contain a JSON object: {path}")
    return manifest


def _manifest_codec_digest(manifest: dict[str, Any]) -> str | None:
    value = _nested(
        manifest,
        "codec_checkpoint_sha256",
        "codec.checkpoint_sha256",
        "codec_identity.checkpoint_sha256",
        "identity.codec_checkpoint_sha256",
        "codec_weights_sha256",
    )
    return str(value).lower() if value else None


def _codec_metadata(checkpoint: Path) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("metadata"), dict):
        raise ValueError(f"Codec checkpoint has no metadata identity: {checkpoint}")
    metadata = payload["metadata"]
    required = ("codec_weights_sha256", "posterior_mode", "codec_source_revision")
    missing = [key for key in required if not metadata.get(key)]
    if missing:
        raise ValueError(f"Codec checkpoint metadata is missing identity fields: {missing}")
    return metadata


def _producer_content_hash(codec_metadata: dict[str, Any], split: str) -> str:
    """Reproduce cache_latents.py's separated-field v1 cache identity."""
    digest = hashlib.sha256()
    fields = (
        ("weights_sha256", codec_metadata["codec_weights_sha256"]),
        ("source_revision", codec_metadata["codec_source_revision"]),
        ("split", split),
        ("posterior_mode", codec_metadata["posterior_mode"]),
        ("normalization_schema", "v1_mean_std_frozen_train"),
        ("preprocessing_schema", "celeba_center_crop_178_resize_64_norm_m1p1"),
        ("latent_channels", 4),
        ("spatial_factor", 4),
        ("pixel_size", 64),
    )
    for label, value in fields:
        digest.update(
            label.encode("utf-8")
            + b"\x00"
            + str(value).encode("utf-8")
            + b"\x00"
        )
    return digest.hexdigest()


def resolve_cache_manifest(cfg: DatasetConfig, split: str) -> tuple[Path, dict[str, Any]]:
    """Resolve exactly one cache by split and codec weight/content identity."""
    split = _canonical_split(split)
    cache_dir = Path(cfg.cache_dir)
    checkpoint = Path(cfg.codec_checkpoint)
    if not cache_dir.exists():
        raise FileNotFoundError(f"CelebA latent cache directory not found: {cache_dir}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Codec checkpoint not found: {checkpoint}")
    codec_metadata = _codec_metadata(checkpoint)
    codec_digest = str(codec_metadata["codec_weights_sha256"]).lower()

    matching: list[tuple[Path, dict[str, Any]]] = []
    examined = 0
    for path in _manifest_candidates(cache_dir):
        manifest = _load_manifest(path)
        examined += 1
        manifest_splits = manifest.get("splits")
        if manifest_splits is None:
            single_split = _nested(manifest, "split", "identity.split", "dataset.split")
            manifest_splits = [single_split] if single_split is not None else []
        manifest_codec_digest = _manifest_codec_digest(manifest)
        if not isinstance(manifest_splits, list) or manifest_codec_digest is None:
            continue
        canonical_splits = []
        for candidate_split in manifest_splits:
            try:
                canonical_splits.append(_canonical_split(str(candidate_split)))
            except ValueError:
                pass
        split_set = set(canonical_splits)
        if split_set == {"train", "valid"}:
            cache_split = "both"
        elif split_set == {"train"}:
            cache_split = "train"
        elif split_set == {"valid"}:
            cache_split = "valid"
        else:
            continue
        expected_content_hash = _producer_content_hash(codec_metadata, cache_split)
        content_hash = manifest.get("content_hash")
        content_matches = content_hash == expected_content_hash
        source_matches = all(
            manifest.get(key) == codec_metadata.get(key)
            for key in ("codec_source", "codec_source_revision", "posterior_mode")
        )
        if (
            split in canonical_splits
            and manifest_codec_digest == codec_digest
            and content_matches
            and source_matches
        ):
            matching.append((path, manifest))

    if not matching:
        raise FileNotFoundError(
            f"No verified latent cache matches split={split!r}, codec weight SHA-256 "
            f"{codec_digest}, and the v1 content identity below {cache_dir} "
            f"({examined} manifests examined)."
        )
    if len(matching) != 1:
        paths = ", ".join(str(item[0]) for item in matching)
        raise RuntimeError(f"Ambiguous latent cache identity; matched {len(matching)} manifests: {paths}")
    return matching[0]


def _validate_manifest_metadata(manifest: dict[str, Any], split: str) -> None:
    expected: tuple[tuple[tuple[str, ...], Any, str], ...] = (
        (("posterior_mode", "codec.posterior_mode"), "mean", "posterior_mode"),
        (("latent_channels", "shape.channels", "codec.latent_channels"), 4, "latent_channels"),
        (("spatial_factor", "codec.spatial_factor"), 4, "spatial_factor"),
        (("pixel_size", "codec.pixel_size"), 64, "pixel_size"),
    )
    for paths, wanted, label in expected:
        got = _nested(manifest, *paths)
        if got is None:
            raise ValueError(f"Latent-cache manifest is missing required {label!r} metadata")
        if str(got).lower() != str(wanted).lower():
            raise ValueError(f"Latent-cache {label} mismatch: expected {wanted!r}, got {got!r}")
    recorded_splits = manifest.get("splits")
    if recorded_splits is None:
        recorded_splits = [_nested(manifest, "split", "identity.split", "dataset.split")]
    if split not in {_canonical_split(str(item)) for item in recorded_splits if item is not None}:
        raise ValueError(f"Latent-cache split metadata does not match requested {split!r}")

    normalization = _nested(manifest, "normalization", "normalization_schema")
    if normalization != "v1_mean_std_frozen_train":
        raise ValueError(f"Unsupported latent normalization schema: {normalization!r}")
    preprocessing = _nested(manifest, "preprocessing", "preprocessing_identity", "preprocessing_schema")
    if preprocessing != "celeba_center_crop_178_resize_64_norm_m1p1":
        raise ValueError(f"Unsupported CelebA preprocessing identity: {preprocessing!r}")


def _load_tensor(path: Path) -> torch.Tensor:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except (TypeError, RuntimeError) as exc:
        warnings.warn(
            f"Read-only mmap is unavailable for {path.name}; using compatible CPU load ({exc}).",
            RuntimeWarning,
            stacklevel=2,
        )
        payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, torch.Tensor):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("latents"), torch.Tensor):
        return payload["latents"]
    raise TypeError(f"Latent tensor file {path} must contain a Tensor or a 'latents' Tensor")


class CelebALatentDataset(Dataset):
    def __init__(self, cfg: DatasetConfig, split: str | None = None):
        self.split = _canonical_split(split or cfg.split or "train")
        manifest_path, self.manifest = resolve_cache_manifest(cfg, self.split)
        _validate_manifest_metadata(self.manifest, self.split)

        filename = f"latents_{self.split}.pt"
        file_hashes = self.manifest.get("file_hashes")
        if isinstance(file_hashes, dict) and filename in file_hashes:
            expected_hash = file_hashes[filename]
        else:
            filename = _nested(self.manifest, "tensor_file", "file", "tensor.path")
            expected_hash = _nested(
                self.manifest, "tensor_sha256", "sha256", "tensor.sha256", "file_sha256"
            )
        if not filename or not expected_hash:
            raise ValueError("Latent-cache manifest must include tensor_file and full tensor_sha256")
        tensor_path = (manifest_path.parent / str(filename)).resolve()
        if not tensor_path.is_file():
            raise FileNotFoundError(f"Latent tensor file named by manifest is missing: {tensor_path}")
        actual_hash = sha256_file(tensor_path)
        if actual_hash != str(expected_hash).lower():
            raise ValueError(
                f"Latent tensor SHA-256 mismatch for {tensor_path}: "
                f"expected {expected_hash}, got {actual_hash}"
            )

        latents = _load_tensor(tensor_path)
        if latents.dtype != torch.float32:
            raise TypeError(f"Latent cache must be float32, got {latents.dtype}")
        if latents.ndim != 4 or tuple(latents.shape[1:]) != EXPECTED_SHAPE:
            raise ValueError(
                f"Latent cache must have shape (N,4,16,16), got {tuple(latents.shape)}"
            )
        if latents.shape[0] < 1 or not torch.isfinite(latents).all():
            raise ValueError("Latent cache must be non-empty and contain only finite values")
        count = _nested(self.manifest, "count", "num_samples", "tensor.count")
        if count is not None and int(count) != latents.shape[0]:
            raise ValueError(
                f"Latent-cache sample count mismatch: manifest={count}, tensor={latents.shape[0]}"
            )
        self.latents = latents

    def __len__(self) -> int:
        return self.latents.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.latents[index], -1


def get_dataloaders(cfg: DatasetConfig, batch_size: int, seed: int):
    train_set = CelebALatentDataset(cfg, split="train")
    valid_set = CelebALatentDataset(cfg, split="valid")
    generator = torch.Generator().manual_seed(seed)
    common = {"batch_size": batch_size, "num_workers": cfg.num_workers, "pin_memory": True}
    train_loader = DataLoader(
        train_set, shuffle=True, drop_last=True, generator=generator, **common
    )
    valid_loader = DataLoader(valid_set, shuffle=False, drop_last=False, **common)
    return train_loader, valid_loader
