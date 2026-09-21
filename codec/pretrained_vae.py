"""Frozen face-specific CompVis CelebA-HQ VQ-f4 codec.

The primary codec maps 64x64 RGB images to deterministic, quantized
three-channel 16x16 latents. The encoder, codebook, and decoder are frozen;
only separately initialized generative U-Nets are trained downstream.
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

import torch

from codec.base import (
    CHECKPOINT_SCHEMA_VERSION,
    REQUIRED_LATENT_CHANNELS,
    REQUIRED_PIXEL_SIZE,
    REQUIRED_SPATIAL_FACTOR,
    BaseCodec,
    CodecCheckpointError,
    validate_checkpoint_metadata,
)

try:
    from diffusers import VQModel as _VQModel
    _DIFFUSERS_AVAILABLE = True
except ImportError:
    _VQModel = None  # type: ignore[assignment]
    _DIFFUSERS_AVAILABLE = False


_CODEC_SOURCE = "CompVis/ldm-celebahq-256"
_CODEC_TYPE = "pretrained_vq_f4"
_DEFAULT_NATIVE_SCALING_FACTOR = 1.0


def _require_diffusers() -> None:
    if not _DIFFUSERS_AVAILABLE:
        raise ImportError(
            "The 'diffusers' package is required for PretrainedVQCodec. "
            "Install the project requirements first."
        )


class PretrainedVQCodec(BaseCodec):
    """Frozen CompVis VQ-f4 encoder/codebook/decoder."""

    latent_channels = 3
    spatial_factor = 4
    pixel_size = 64
    posterior_mode = "quantized"

    def __init__(
        self,
        vae,
        native_scaling_factor: float,
        codec_source: str,
        codec_source_revision: str,
        codec_weights_sha256: str,
        source_path: str,
        device: torch.device,
    ) -> None:
        super().__init__()
        _require_diffusers()
        if not source_path:
            raise ValueError("source_path must be a non-empty local directory path")
        self._vae = vae
        self._native_scaling_factor = float(native_scaling_factor)
        self._codec_source = codec_source
        self._codec_source_revision = codec_source_revision
        self._codec_weights_sha256 = codec_weights_sha256
        self._source_path = os.path.abspath(source_path)
        self._device = device
        self._vae.eval().requires_grad_(False).to(device)
        self._verify_shape()

    @classmethod
    def from_pretrained(
        cls,
        source_path: str,
        device: torch.device,
        codec_source: str = _CODEC_SOURCE,
        codec_source_revision: str = "local",
        native_scaling_factor: float = _DEFAULT_NATIVE_SCALING_FACTOR,
    ) -> "PretrainedVQCodec":
        _require_diffusers()
        if not os.path.isdir(source_path):
            raise FileNotFoundError(
                f"Pretrained VQ source directory not found: '{source_path}'. "
                "Download CompVis/ldm-celebahq-256 vqvae files first."
            )
        vae = _VQModel.from_pretrained(source_path)
        return cls(
            vae=vae,
            native_scaling_factor=native_scaling_factor,
            codec_source=codec_source,
            codec_source_revision=codec_source_revision,
            codec_weights_sha256=cls._weights_sha256(vae),
            source_path=source_path,
            device=device,
        )

    @classmethod
    def from_checkpoint(
        cls, path: str, device: torch.device, require_frozen: bool = True
    ) -> "PretrainedVQCodec":
        _require_diffusers()
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        meta = ckpt.get("metadata", {})
        validate_checkpoint_metadata(meta, path=path, require_frozen=require_frozen)
        if meta.get("codec_type") != _CODEC_TYPE:
            raise CodecCheckpointError(
                f"Checkpoint '{path}' has codec_type={meta.get('codec_type')!r}; "
                f"expected {_CODEC_TYPE!r}."
            )
        source_path = ckpt.get("source_path")
        if not source_path or not os.path.isdir(source_path):
            raise CodecCheckpointError(
                f"Checkpoint '{path}' records unavailable source_path={source_path!r}."
            )
        vae = _VQModel.from_pretrained(source_path)
        actual_sha = cls._weights_sha256(vae)
        stored_sha = meta["codec_weights_sha256"]
        if actual_sha != stored_sha:
            raise CodecCheckpointError(
                f"Codec weight digest mismatch for '{path}': stored={stored_sha}, "
                f"actual={actual_sha}. Revalidate the codec."
            )
        codec = cls(
            vae=vae,
            native_scaling_factor=meta["native_scaling_factor"],
            codec_source=meta["codec_source"],
            codec_source_revision=meta["codec_source_revision"],
            codec_weights_sha256=stored_sha,
            source_path=source_path,
            device=device,
        )
        if meta.get("stats_frozen") is True:
            codec._load_frozen_stats(meta["latent_mean"], meta["latent_std"], device)
        return codec

    @staticmethod
    def _weights_sha256(vae) -> str:
        state = {f"vqvae.{key}": value for key, value in vae.state_dict().items()}
        return BaseCodec._sha256_of_state_dict(state)

    @torch.no_grad()
    def _encode_quantized(self, images: torch.Tensor) -> torch.Tensor:
        encoded = self._vae.encode(images).latents
        quantized, _, _ = self._vae.quantize(encoded)
        return quantized

    def _verify_shape(self) -> None:
        with torch.no_grad():
            dummy = torch.zeros(1, 3, REQUIRED_PIXEL_SIZE, REQUIRED_PIXEL_SIZE,
                                device=self._device)
            try:
                z = self._encode_quantized(dummy)
            except Exception as exc:
                raise CodecCheckpointError(
                    f"VQ encode/quantize failed on synthetic 64x64 input: {exc}"
                ) from exc
        expected_size = REQUIRED_PIXEL_SIZE // REQUIRED_SPATIAL_FACTOR
        expected = (1, REQUIRED_LATENT_CHANNELS, expected_size, expected_size)
        if tuple(z.shape) != expected:
            raise CodecCheckpointError(
                f"VQ-f4 shape verification failed: expected {expected}, "
                f"got {tuple(z.shape)}."
            )

    @torch.no_grad()
    def encode_mean(self, images: torch.Tensor) -> torch.Tensor:
        """Return deterministic quantized VQ latents (legacy API method name)."""
        self._assert_pixel_input(images)
        z = self._encode_quantized(images.to(self._device))
        return z * self._native_scaling_factor

    @torch.no_grad()
    def decode(self, scaled_latents: torch.Tensor) -> torch.Tensor:
        self._assert_latent_input(scaled_latents, name="scaled_latents")
        z = scaled_latents.to(self._device) / self._native_scaling_factor
        return self._vae.decode(z, force_not_quantize=False).sample.clamp(-1.0, 1.0)

    def save(
        self,
        path: str,
        *,
        validation_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not self._stats_frozen:
            raise RuntimeError(
                "Cannot save codec before training-set statistics are frozen."
            )
        meta = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "codec_type": _CODEC_TYPE,
            "codec_source": self._codec_source,
            "codec_source_revision": self._codec_source_revision,
            "codec_weights_sha256": self._codec_weights_sha256,
            "latent_channels": self.latent_channels,
            "spatial_factor": self.spatial_factor,
            "pixel_size": self.pixel_size,
            "posterior_mode": self.posterior_mode,
            "native_scaling_factor": self._native_scaling_factor,
            "stats_frozen": True,
            "latent_mean": self._stats_to_list(self.latent_mean),
            "latent_std": self._stats_to_list(self.latent_std),
            "dataset": "celeba",
            "image_size": self.pixel_size,
        }
        validate_checkpoint_metadata(meta, path=path, require_frozen=True)
        payload: dict[str, Any] = {
            "metadata": meta,
            "source_path": self._find_source_path(),
        }
        if validation_metadata is not None:
            payload["validation"] = dict(validation_metadata)
        _atomic_save(payload, path)

    def _find_source_path(self) -> str:
        source_path = getattr(self, "_source_path", None)
        if not source_path:
            raise RuntimeError("Cannot determine source_path for codec checkpoint")
        return source_path


def _atomic_save(payload: dict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = f"{path}.tmp-{os.getpid()}"
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
