"""
tests/test_codec_base.py — CPU-only unit tests for codec/base.py.

Tests cover:
  - validate_checkpoint_metadata(): all rejection paths and happy path
  - BaseCodec: normalization round-trip with synthetic stats
  - BaseCodec: compute_and_freeze_stats() with a tiny synthetic dataloader
  - BaseCodec: double-freeze guard
  - SHA-256 helper: stable ordering
"""
import pytest
import torch
from torch.utils.data import TensorDataset, DataLoader

from codec.base import (
    BaseCodec,
    CodecCheckpointError,
    CHECKPOINT_SCHEMA_VERSION,
    REQUIRED_LATENT_CHANNELS,
    REQUIRED_PIXEL_SIZE,
    REQUIRED_SPATIAL_FACTOR,
    validate_checkpoint_metadata,
)

LATENT_SIZE = REQUIRED_PIXEL_SIZE // REQUIRED_SPATIAL_FACTOR


# ── Minimal concrete subclass for testing ────────────────────────────────────
class _MockCodec(BaseCodec):
    """Concrete codec that returns deterministic zero-tensors."""

    def encode_mean(self, images: torch.Tensor) -> torch.Tensor:
        B = images.shape[0]
        return torch.zeros(B, REQUIRED_LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        return torch.zeros(B, 3, REQUIRED_PIXEL_SIZE, REQUIRED_PIXEL_SIZE)

    def save(self, path: str) -> None:
        pass  # no-op for testing


class _NonConstantCodec(BaseCodec):
    """
    Concrete codec with non-constant encode_mean so std > 0.
    Each image produces latents that vary across the batch dimension,
    which means compute_and_freeze_stats won't hit the zero-std guard.
    """

    def encode_mean(self, images: torch.Tensor) -> torch.Tensor:
        B = images.shape[0]
        # Use the mean of each image's pixels to produce varied latents.
        # images: (B, 3, 64, 64) → reduce to scalar per image then expand.
        base = images.float().mean(dim=(1, 2, 3), keepdim=True)  # (B, 1, 1, 1)
        base = base.expand(B, 1, 1, 1)
        # Produce 4 channels with different offsets so each channel has variance.
        offsets = torch.tensor([0.0, 0.1, -0.1, 0.2], device=images.device)
        z = base.expand(B, 4, 1, 1) + offsets.view(1, 4, 1, 1)
        return z.expand(B, 4, LATENT_SIZE, LATENT_SIZE).contiguous()

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        return torch.zeros(B, 3, REQUIRED_PIXEL_SIZE, REQUIRED_PIXEL_SIZE)

    def save(self, path: str) -> None:
        pass



# ── Fixtures ──────────────────────────────────────────────────────────────────
def _valid_meta(**overrides) -> dict:
    base = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "codec_type": "pretrained_kl_vae",
        "codec_source": "stabilityai/sd-vae-ft-mse",
        "codec_source_revision": "abc123",
        "codec_weights_sha256": "deadbeef",
        "latent_channels": REQUIRED_LATENT_CHANNELS,
        "spatial_factor": REQUIRED_SPATIAL_FACTOR,
        "pixel_size": REQUIRED_PIXEL_SIZE,
        "posterior_mode": "mean",
        "native_scaling_factor": 1.0,
        "stats_frozen": True,
        "latent_mean": [[[[0.1]], [[0.2]], [[-0.3]], [[0.4]]]],
        "latent_std":  [[[[0.9]], [[0.8]],  [[0.7]], [[0.6]]]],
        "dataset": "celeba",
        "image_size": 64,
    }
    base.update(overrides)
    return base


# ══════════════════════════════════════════════════════════════════════════════
# validate_checkpoint_metadata
# ══════════════════════════════════════════════════════════════════════════════

class TestValidateCheckpointMetadata:

    def test_valid_metadata_passes(self):
        validate_checkpoint_metadata(_valid_meta())  # must not raise

    def test_rejects_missing_schema_version(self):
        meta = _valid_meta()
        del meta["schema_version"]
        with pytest.raises(CodecCheckpointError, match="schema_version"):
            validate_checkpoint_metadata(meta)

    def test_rejects_unsupported_schema_version(self):
        with pytest.raises(CodecCheckpointError, match="schema_version"):
            validate_checkpoint_metadata(_valid_meta(schema_version=99))

    def test_rejects_missing_codec_type(self):
        meta = _valid_meta()
        del meta["codec_type"]
        with pytest.raises(CodecCheckpointError, match="codec_type"):
            validate_checkpoint_metadata(meta)

    def test_rejects_empty_codec_source(self):
        with pytest.raises(CodecCheckpointError, match="codec_source"):
            validate_checkpoint_metadata(_valid_meta(codec_source=""))

    def test_rejects_unsupported_posterior_mode(self):
        with pytest.raises(CodecCheckpointError, match="posterior_mode"):
            validate_checkpoint_metadata(_valid_meta(posterior_mode="sample"))

    def test_rejects_wrong_latent_channels(self):
        with pytest.raises(CodecCheckpointError, match="latent_channels"):
            validate_checkpoint_metadata(_valid_meta(latent_channels=8))

    def test_accepts_primary_factor8(self):
        validate_checkpoint_metadata(_valid_meta(spatial_factor=8))

    def test_accepts_historical_scratch_factor4(self):
        validate_checkpoint_metadata(_valid_meta(spatial_factor=4))

    def test_rejects_unsupported_factor(self):
        with pytest.raises(CodecCheckpointError, match="spatial_factor"):
            validate_checkpoint_metadata(_valid_meta(spatial_factor=2))

    def test_rejects_wrong_pixel_size(self):
        with pytest.raises(CodecCheckpointError, match="pixel_size"):
            validate_checkpoint_metadata(_valid_meta(pixel_size=32))

    def test_rejects_non_positive_native_scaling_factor(self):
        with pytest.raises(CodecCheckpointError, match="native_scaling_factor"):
            validate_checkpoint_metadata(_valid_meta(native_scaling_factor=-1.0))

    def test_rejects_zero_native_scaling_factor(self):
        with pytest.raises(CodecCheckpointError, match="native_scaling_factor"):
            validate_checkpoint_metadata(_valid_meta(native_scaling_factor=0.0))

    def test_rejects_unfrozen_stats_when_required(self):
        meta = _valid_meta(stats_frozen=False)
        meta["latent_mean"] = None
        meta["latent_std"] = None
        with pytest.raises(CodecCheckpointError, match="stats_frozen"):
            validate_checkpoint_metadata(meta, require_frozen=True)

    def test_allows_unfrozen_stats_when_not_required(self):
        meta = _valid_meta(stats_frozen=False)
        meta["latent_mean"] = None
        meta["latent_std"] = None
        # Must not raise when require_frozen=False
        validate_checkpoint_metadata(meta, require_frozen=False)

    def test_rejects_wrong_latent_mean_shape(self):
        meta = _valid_meta()
        meta["latent_mean"] = [[[0.1, 0.2]]]  # wrong shape
        with pytest.raises(CodecCheckpointError, match="latent_mean"):
            validate_checkpoint_metadata(meta)

    def test_rejects_nonpositive_latent_std(self):
        meta = _valid_meta()
        meta["latent_std"] = [[[[0.0]], [[0.8]], [[0.7]], [[0.6]]]]
        with pytest.raises(CodecCheckpointError, match="latent_std"):
            validate_checkpoint_metadata(meta)

    def test_rejects_negative_latent_std(self):
        meta = _valid_meta()
        meta["latent_std"] = [[[[-0.1]], [[0.8]], [[0.7]], [[0.6]]]]
        with pytest.raises(CodecCheckpointError, match="latent_std"):
            validate_checkpoint_metadata(meta)

    def test_rejects_missing_image_size(self):
        meta = _valid_meta()
        del meta["image_size"]
        with pytest.raises(CodecCheckpointError, match="image_size"):
            validate_checkpoint_metadata(meta)

    def test_rejects_non_finite_native_scaling_factor(self):
        with pytest.raises(CodecCheckpointError, match="native_scaling_factor"):
            validate_checkpoint_metadata(_valid_meta(native_scaling_factor=float("inf")))


# ══════════════════════════════════════════════════════════════════════════════
# BaseCodec: normalization
# ══════════════════════════════════════════════════════════════════════════════

class TestBaseCodecNormalization:

    def _codec_with_stats(self) -> _MockCodec:
        codec = _MockCodec()
        codec._latent_mean = torch.tensor([[[[0.1]], [[0.2]], [[-0.3]], [[0.4]]]])
        codec._latent_std  = torch.tensor([[[[0.9]], [[0.8]],  [[0.7]], [[0.6]]]])
        codec._stats_frozen = True
        return codec

    def test_normalise_denormalise_roundtrip(self):
        codec = self._codec_with_stats()
        z = torch.randn(4, 4, LATENT_SIZE, LATENT_SIZE)
        z_back = codec.denormalise(codec.normalise(z))
        assert torch.allclose(z, z_back, atol=1e-5), \
            f"Round-trip error: {(z - z_back).abs().max():.2e}"

    def test_normalise_has_approx_zero_mean(self):
        """After normalization, per-channel mean should be ~0 for the reference z."""
        codec = self._codec_with_stats()
        # z constructed to exactly match the codec's mean
        z = codec.latent_mean.expand(100, -1, -1, -1).clone()
        z_norm = codec.normalise(z)
        assert z_norm.abs().max() < 1e-5, \
            f"Normalised z at mean should be ~0, got max={z_norm.abs().max():.2e}"

    def test_decode_normalised_shape(self):
        codec = self._codec_with_stats()
        z_norm = torch.randn(3, 4, LATENT_SIZE, LATENT_SIZE)
        out = codec.decode_normalised(z_norm)
        assert out.shape == (3, 3, REQUIRED_PIXEL_SIZE, REQUIRED_PIXEL_SIZE)

    def test_latent_mean_raises_before_frozen(self):
        codec = _MockCodec()
        with pytest.raises(RuntimeError, match="latent_mean"):
            _ = codec.latent_mean

    def test_latent_std_raises_before_frozen(self):
        codec = _MockCodec()
        with pytest.raises(RuntimeError, match="latent_std"):
            _ = codec.latent_std


# ══════════════════════════════════════════════════════════════════════════════
# BaseCodec: compute_and_freeze_stats
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeAndFreezeStats:

    def _make_loader(self, n_batches: int = 4, batch_size: int = 8):
        """Synthetic CelebA-like loader: (B, 3, 64, 64) images + dummy labels."""
        torch.manual_seed(42)
        # Use randn so pixel means vary across the batch → non-zero latent std
        images = torch.randn(n_batches * batch_size, 3, 64, 64)
        labels = torch.zeros(n_batches * batch_size, dtype=torch.long)
        ds = TensorDataset(images, labels)
        return DataLoader(ds, batch_size=batch_size)

    def test_stats_frozen_after_compute(self):
        # Use NonConstantCodec so std > 0 and compute succeeds.
        codec = _NonConstantCodec()
        loader = self._make_loader()
        assert codec.stats_frozen is False
        codec.compute_and_freeze_stats(loader, device=torch.device("cpu"))
        assert codec.stats_frozen is True

    def test_stats_shape_is_1_4_1_1(self):
        codec = _NonConstantCodec()
        loader = self._make_loader()
        codec.compute_and_freeze_stats(loader, device=torch.device("cpu"))
        assert codec.latent_mean.shape == (1, REQUIRED_LATENT_CHANNELS, 1, 1)
        assert codec.latent_std.shape  == (1, REQUIRED_LATENT_CHANNELS, 1, 1)

    def test_std_is_strictly_positive(self):
        """After compute_and_freeze_stats with varied latents, all stds are > 0."""
        codec = _NonConstantCodec()
        loader = self._make_loader()
        codec.compute_and_freeze_stats(loader, device=torch.device("cpu"))
        assert (codec.latent_std > 0).all(), \
            f"All std values must be > 0, got: {codec.latent_std}"

    def test_zero_std_guard_raises(self):
        """A codec that always returns the same latent should trip the guard."""
        codec = _MockCodec()  # always returns zeros → std = 0
        loader = self._make_loader()
        with pytest.raises(RuntimeError, match="non-positive"):
            codec.compute_and_freeze_stats(loader, device=torch.device("cpu"))

    def test_double_freeze_raises(self):
        codec = _NonConstantCodec()
        loader = self._make_loader()
        codec.compute_and_freeze_stats(loader, device=torch.device("cpu"))
        with pytest.raises(RuntimeError, match="already been called"):
            codec.compute_and_freeze_stats(loader, device=torch.device("cpu"))



# ══════════════════════════════════════════════════════════════════════════════
# SHA-256 helper
# ══════════════════════════════════════════════════════════════════════════════

class TestSHA256Helper:

    def test_stable_across_key_order(self):
        sd1 = {"b": torch.ones(3), "a": torch.zeros(2)}
        sd2 = {"a": torch.zeros(2), "b": torch.ones(3)}
        assert _MockCodec._sha256_of_state_dict(sd1) == \
               _MockCodec._sha256_of_state_dict(sd2)

    def test_changes_with_different_values(self):
        sd1 = {"a": torch.zeros(4)}
        sd2 = {"a": torch.ones(4)}
        assert _MockCodec._sha256_of_state_dict(sd1) != \
               _MockCodec._sha256_of_state_dict(sd2)

    def test_hexdigest_is_64_chars(self):
        sd = {"w": torch.randn(2, 2)}
        digest = _MockCodec._sha256_of_state_dict(sd)
        assert len(digest) == 64
        assert all(c in "0123456789abcdef" for c in digest)
