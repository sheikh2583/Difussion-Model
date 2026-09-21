"""Small factor-4 KL-VAE fallback for CelebA 64x64.

This is intentionally the transparent MSE+KL baseline. It is not a claim that
pixel MSE alone will satisfy the project's strict perceptual acceptance gate.
"""
from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from codec.base import (
    CHECKPOINT_SCHEMA_VERSION,
    BaseCodec,
    CodecCheckpointError,
    validate_checkpoint_metadata,
)


def _group_count(channels: int, maximum: int = 8) -> int:
    """Largest group count <= maximum that divides channels exactly."""
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(torch.nn.functional.silu(self.norm1(inputs)))
        hidden = self.conv2(torch.nn.functional.silu(self.norm2(hidden)))
        return hidden + self.skip(inputs)


class ScratchKLVAE(nn.Module, BaseCodec):
    latent_channels = 4
    spatial_factor = 4
    pixel_size = 64
    posterior_mode = "mean"
    native_scaling_factor = 1.0
    codec_type = "scratch_kl_vae"
    codec_source = "project_scratch_kl_vae"
    codec_source_revision = "architecture-v1"

    def __init__(self) -> None:
        nn.Module.__init__(self)
        BaseCodec.__init__(self)
        self._native_scaling_factor = self.native_scaling_factor
        self._codec_source = self.codec_source
        self._codec_source_revision = self.codec_source_revision
        self._codec_weights_sha256 = ""
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            ResidualBlock(64, 64),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            ResidualBlock(128, 128),
            nn.Conv2d(128, 128, 4, stride=2, padding=1),
            ResidualBlock(128, 128),
            nn.GroupNorm(_group_count(128), 128),
            nn.SiLU(),
            nn.Conv2d(128, 2 * self.latent_channels, 3, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(self.latent_channels, 128, 3, padding=1),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1),
            ResidualBlock(128, 128),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            ResidualBlock(64, 64),
            nn.GroupNorm(_group_count(64), 64),
            nn.SiLU(),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Tanh(),
        )

    def _apply(self, fn):
        """Keep BaseCodec's non-module statistics on the module device."""
        result = super()._apply(fn)
        if self._latent_mean is not None:
            self._latent_mean = fn(self._latent_mean)
        if self._latent_std is not None:
            self._latent_std = fn(self._latent_std)
        return result

    @staticmethod
    def _bounded_logvar(logvar: torch.Tensor) -> torch.Tensor:
        return logvar.clamp(min=-30.0, max=20.0)

    def posterior_parameters(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if images.ndim != 4 or images.shape[1:] != (3, 64, 64):
            raise ValueError(f"Expected images shaped (B,3,64,64), got {tuple(images.shape)}")
        mean, logvar = self.encoder(images).chunk(2, dim=1)
        return mean, self._bounded_logvar(logvar)

    def encode(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return reparameterized sample, posterior mean, and bounded log variance."""
        mean, logvar = self.posterior_parameters(images)
        sample = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)
        return sample, mean, logvar

    def encode_mean(self, images: torch.Tensor) -> torch.Tensor:
        mean, _ = self.posterior_parameters(images)
        return mean * self.native_scaling_factor

    def decode(self, scaled_latents: torch.Tensor) -> torch.Tensor:
        if scaled_latents.ndim != 4 or scaled_latents.shape[1:] != (4, 16, 16):
            raise ValueError(
                f"Expected scaled latents shaped (B,4,16,16), got {tuple(scaled_latents.shape)}"
            )
        return self.decoder(scaled_latents / self.native_scaling_factor)

    @staticmethod
    def kl_loss(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # Keep exp() in fp32 even when the training forward pass uses AMP.
        # exp(20) is finite in fp32 but overflows float16, which can otherwise
        # turn an otherwise healthy mixed-precision run into NaNs.
        mean_fp32 = mean.float()
        bounded = ScratchKLVAE._bounded_logvar(logvar.float())
        return 0.5 * (
            mean_fp32.square() + bounded.exp() - 1.0 - bounded
        ).mean()

    @torch.no_grad()
    def compute_and_freeze_stats(
        self, dataloader: Iterable[Any], device: torch.device | str
    ) -> None:
        """Compute exact per-channel moments over all train images/spatial sites."""
        if self._stats_frozen:
            raise RuntimeError("Latent statistics are already frozen")
        was_training = self.training
        self.eval().to(device)
        channel_sum = torch.zeros(4, dtype=torch.float64, device=device)
        channel_sq_sum = torch.zeros_like(channel_sum)
        element_count = 0
        for item in dataloader:
            images = item[0] if isinstance(item, (tuple, list)) else item
            latents = self.encode_mean(images.to(device, non_blocking=True)).double()
            channel_sum += latents.sum(dim=(0, 2, 3))
            channel_sq_sum += latents.square().sum(dim=(0, 2, 3))
            element_count += latents.shape[0] * latents.shape[2] * latents.shape[3]
        if element_count < 2:
            raise ValueError("At least two latent elements are required to freeze statistics")
        mean = channel_sum / element_count
        variance = (channel_sq_sum / element_count - mean.square()).clamp_min(0.0)
        std = variance.sqrt()
        if not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError(f"Cannot freeze degenerate latent statistics: std={std.tolist()}")
        self._latent_mean = mean.float().view(1, 4, 1, 1)
        self._latent_std = std.float().view(1, 4, 1, 1)
        self._stats_frozen = True
        self.train(was_training)

    def _weights_sha256(self) -> str:
        return self._sha256_of_state_dict(self.state_dict())

    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "codec_type": self.codec_type,
            "codec_source": self.codec_source,
            "codec_source_revision": self.codec_source_revision,
            "codec_weights_sha256": self._weights_sha256(),
            "latent_channels": self.latent_channels,
            "spatial_factor": self.spatial_factor,
            "pixel_size": self.pixel_size,
            "posterior_mode": self.posterior_mode,
            "native_scaling_factor": self.native_scaling_factor,
            "stats_frozen": self.stats_frozen,
            "latent_mean": self._stats_to_list(self.latent_mean),
            "latent_std": self._stats_to_list(self.latent_std),
            "dataset": "celeba",
            "image_size": 64,
        }

    def save(self, path: str | os.PathLike[str]) -> None:
        if not self.stats_frozen:
            raise RuntimeError("Cannot save scratch codec before statistics are frozen")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f"{target.name}.tmp-{os.getpid()}")
        metadata = self.metadata()
        self._codec_weights_sha256 = metadata["codec_weights_sha256"]
        validate_checkpoint_metadata(metadata, path=str(target), require_frozen=True)
        payload = {"metadata": metadata, "model_state_dict": self.state_dict()}
        try:
            with temporary.open("wb") as handle:
                torch.save(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def from_checkpoint(
        cls,
        path: str | os.PathLike[str],
        device: torch.device | str = "cpu",
        require_frozen: bool = True,
    ) -> ScratchKLVAE:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or not {"metadata", "model_state_dict"} <= payload.keys():
            raise CodecCheckpointError(
                "Scratch codec checkpoint must contain metadata and model_state_dict"
            )
        metadata = payload["metadata"]
        validate_checkpoint_metadata(
            metadata, path=str(path), require_frozen=require_frozen
        )
        if metadata.get("codec_type") != cls.codec_type:
            raise CodecCheckpointError(
                f"Checkpoint codec_type={metadata.get('codec_type')!r}; "
                f"expected {cls.codec_type!r}"
            )
        model = cls()
        model.load_state_dict(payload["model_state_dict"], strict=True)
        if metadata.get("codec_weights_sha256") != model._weights_sha256():
            raise CodecCheckpointError(
                "Scratch codec weight digest does not match checkpoint metadata"
            )
        model._codec_weights_sha256 = metadata["codec_weights_sha256"]
        model._codec_source = metadata["codec_source"]
        model._codec_source_revision = metadata["codec_source_revision"]
        model._native_scaling_factor = float(metadata["native_scaling_factor"])
        if metadata.get("stats_frozen") is True:
            model._load_frozen_stats(
                metadata["latent_mean"], metadata["latent_std"], torch.device(device)
            )
        return model.to(device)

    load = from_checkpoint
