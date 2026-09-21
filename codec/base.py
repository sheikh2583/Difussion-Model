"""
codec/base.py — shared codec contract for both the pretrained and scratch backends.

All codec backends (PretrainedKLVAE, ScratchKLVAE) inherit from BaseCodec and
expose the exact same public API so the rest of the pipeline (cache_latents,
evaluator, sampler, smoke test) never needs to know which backend is active.

Checkpoint schema (schema_version=1)
─────────────────────────────────────
Required keys in every saved checkpoint dict:

    schema_version          int   must equal CHECKPOINT_SCHEMA_VERSION (1)
    codec_type              str   "pretrained_kl_vae" | "scratch_kl_vae"
    codec_source            str   human-readable origin (HuggingFace repo id, path, …)
    codec_source_revision   str   git commit, tag, or content hash
    codec_weights_sha256    str   hex SHA-256 of the serialised encoder+decoder weights
    latent_channels         int   must equal REQUIRED_LATENT_CHANNELS (4)
    spatial_factor          int   must equal REQUIRED_SPATIAL_FACTOR (4)
    pixel_size              int   must equal REQUIRED_PIXEL_SIZE (64)
    posterior_mode          str   "mean"
    native_scaling_factor   float positive finite scalar
    stats_frozen            bool  True means latent_mean/latent_std are training-set stats
    latent_mean             list  shape (1, 4, 1, 1) — serialised as nested list
    latent_std              list  shape (1, 4, 1, 1) — serialised as nested list; all > 0
    dataset                 str   dataset the statistics were frozen on (e.g. "celeba")
    image_size              int   pixel dimension the codec was calibrated on

Shape / range conventions
──────────────────────────
    pixel input     : (B, 3, 64, 64)  in [-1, 1]
    native latent   : (B, 4, 16, 16)  in the codec's own scaling convention
    normalised latent: (B, 4, 16, 16) approximately zero-mean / unit std per channel

Normalization is frozen once compute_and_freeze_stats() has been called on the
training-set distribution.  Subsequent inference callers must never recompute it.
"""
import hashlib
from abc import ABC, abstractmethod
from typing import Any, Dict

import torch
import torch.nn as nn


# ── Hard-wired experiment constants ──────────────────────────────────────────
CHECKPOINT_SCHEMA_VERSION: int = 1
REQUIRED_LATENT_CHANNELS: int = 4
REQUIRED_SPATIAL_FACTOR: int = 4      # 64 → 16
REQUIRED_PIXEL_SIZE: int = 64

_SUPPORTED_SCHEMA_VERSIONS = frozenset({1})
_SUPPORTED_POSTERIOR_MODES = frozenset({"mean"})


# ── Exception ─────────────────────────────────────────────────────────────────
class CodecCheckpointError(RuntimeError):
    """
    Raised when a codec checkpoint fails structural or metadata validation.
    Always includes the offending path and a precise failure description so
    the operator knows exactly what to fix before retrying.
    """


# ── Metadata validation (free function, usable before constructing a codec) ──
def validate_checkpoint_metadata(
    meta: Dict[str, Any],
    path: str = "<unknown>",
    *,
    require_frozen: bool = True,
) -> None:
    """
    Validate a codec checkpoint metadata dict against the schema contract.

    Parameters
    ----------
    meta            : dict loaded from a codec checkpoint (or equivalent)
    path            : string for error messages only (file path or description)
    require_frozen  : if True (default), reject checkpoints where stats_frozen != True.
                      Pass False only during the initial stat-computation phase
                      (validate_codec.py) before the stats have been frozen.

    Raises
    ------
    CodecCheckpointError on any validation failure.
    """
    def _fail(msg: str) -> None:
        raise CodecCheckpointError(
            f"Codec checkpoint validation failed for '{path}': {msg}"
        )

    # ── Schema version ────────────────────────────────────────────────────────
    sv = meta.get("schema_version")
    if sv is None:
        _fail("missing 'schema_version' key")
    if sv not in _SUPPORTED_SCHEMA_VERSIONS:
        _fail(
            f"unsupported schema_version={sv!r}; "
            f"supported: {sorted(_SUPPORTED_SCHEMA_VERSIONS)}"
        )

    # ── Required string keys ──────────────────────────────────────────────────
    for key in ("codec_type", "codec_source", "codec_source_revision",
                "codec_weights_sha256", "posterior_mode", "dataset"):
        if not isinstance(meta.get(key), str) or not meta[key]:
            _fail(f"missing or empty string field '{key}'")

    # ── Posterior mode ────────────────────────────────────────────────────────
    pm = meta["posterior_mode"]
    if pm not in _SUPPORTED_POSTERIOR_MODES:
        _fail(f"unsupported posterior_mode={pm!r}; supported: {_SUPPORTED_POSTERIOR_MODES}")

    # ── Latent shape parameters ───────────────────────────────────────────────
    lc = meta.get("latent_channels")
    sf = meta.get("spatial_factor")
    ps = meta.get("pixel_size")
    if lc != REQUIRED_LATENT_CHANNELS:
        _fail(
            f"latent_channels={lc!r} but experiment requires {REQUIRED_LATENT_CHANNELS}. "
            f"This codec produces the wrong latent shape."
        )
    if sf != REQUIRED_SPATIAL_FACTOR:
        _fail(
            f"spatial_factor={sf!r} but experiment requires {REQUIRED_SPATIAL_FACTOR}. "
            f"Note: SD 1.x VAE is normally factor-8 (8×8 latents from 64×64 input) "
            f"and must NOT be used here — use a genuine factor-4 KL checkpoint."
        )
    if ps != REQUIRED_PIXEL_SIZE:
        _fail(
            f"pixel_size={ps!r} but experiment requires {REQUIRED_PIXEL_SIZE}."
        )

    # ── Native scaling factor ─────────────────────────────────────────────────
    nsf = meta.get("native_scaling_factor")
    if not isinstance(nsf, (int, float)):
        _fail("missing or non-numeric 'native_scaling_factor'")
    if not (torch.tensor(float(nsf)).isfinite().item() and float(nsf) > 0):
        _fail(f"native_scaling_factor={nsf!r} must be positive and finite")

    # ── Frozen-stats requirement ──────────────────────────────────────────────
    stats_frozen = meta.get("stats_frozen")
    if require_frozen and stats_frozen is not True:
        _fail(
            f"stats_frozen={stats_frozen!r} but require_frozen=True. "
            f"Run validate_codec.py to compute and freeze training-set statistics "
            f"before using this codec for inference or caching."
        )

    if stats_frozen is True:
        # Validate the actual stat tensors
        for stat_key in ("latent_mean", "latent_std"):
            raw = meta.get(stat_key)
            if raw is None:
                _fail(f"stats_frozen=True but '{stat_key}' is missing")
            try:
                t = torch.tensor(raw, dtype=torch.float32)
            except Exception as exc:
                _fail(f"cannot deserialise '{stat_key}': {exc}")
            if t.shape != (1, REQUIRED_LATENT_CHANNELS, 1, 1):
                _fail(
                    f"'{stat_key}' has shape {tuple(t.shape)} but expected "
                    f"(1, {REQUIRED_LATENT_CHANNELS}, 1, 1)"
                )
            if not t.isfinite().all().item():
                _fail(f"'{stat_key}' contains non-finite values")

        # Standard deviations must be strictly positive
        std_t = torch.tensor(meta["latent_std"], dtype=torch.float32)
        if not (std_t > 0).all().item():
            _fail(
                f"'latent_std' contains non-positive values: {std_t.flatten().tolist()}. "
                f"All channel standard deviations must be strictly positive."
            )

    # ── image_size ────────────────────────────────────────────────────────────
    img_sz = meta.get("image_size")
    if not isinstance(img_sz, int) or img_sz <= 0:
        _fail(f"missing or invalid 'image_size' (got {img_sz!r})")


# ── Abstract base class ───────────────────────────────────────────────────────
class BaseCodec(ABC):
    """
    Common interface implemented by both PretrainedKLVAE and ScratchKLVAE.

    Conventions:
        • pixel input  : (B, 3, 64, 64) float32 in [-1, 1]
        • native latent: (B, 4, 16, 16) float32 in the codec's own scale
        • normalised   : (B, 4, 16, 16) float32 ≈ N(0,1) per channel

    All tensors returned by methods in this class live on the same device
    as the codec.  Callers should not assume any particular device.
    """

    # ── Intrinsic properties (set by subclasses) ──────────────────────────────
    latent_channels: int = REQUIRED_LATENT_CHANNELS
    spatial_factor: int = REQUIRED_SPATIAL_FACTOR
    pixel_size: int = REQUIRED_PIXEL_SIZE
    posterior_mode: str = "mean"

    def __init__(self) -> None:
        # Normalization statistics — None until compute_and_freeze_stats() is called.
        self._latent_mean: torch.Tensor | None = None
        self._latent_std: torch.Tensor | None = None
        self._stats_frozen: bool = False

    # ── Frozen-stat properties ────────────────────────────────────────────────
    @property
    def stats_frozen(self) -> bool:
        return self._stats_frozen

    @property
    def latent_mean(self) -> torch.Tensor:
        if self._latent_mean is None:
            raise RuntimeError(
                "latent_mean has not been set. Call compute_and_freeze_stats() "
                "on the training set first, then save and reload the codec."
            )
        return self._latent_mean

    @property
    def latent_std(self) -> torch.Tensor:
        if self._latent_std is None:
            raise RuntimeError(
                "latent_std has not been set. Call compute_and_freeze_stats() "
                "on the training set first, then save and reload the codec."
            )
        return self._latent_std

    # ── Core encoding / decoding (must be implemented) ───────────────────────
    @abstractmethod
    def encode_mean(self, images: torch.Tensor) -> torch.Tensor:
        """
        Deterministic posterior-mean encoding.

        Parameters
        ----------
        images : (B, 3, H, W) float32 in [-1, 1]

        Returns
        -------
        (B, 4, H//spatial_factor, W//spatial_factor) float32
            Native scaled latents (before normalization).
        """
        raise NotImplementedError

    @abstractmethod
    def decode(self, scaled_latents: torch.Tensor) -> torch.Tensor:
        """
        Decode native-scaled latents back to pixel space.

        Parameters
        ----------
        scaled_latents : (B, 4, H', W') float32 in the codec's native scale

        Returns
        -------
        (B, 3, H, W) float32 in [-1, 1]
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Persist codec identity and frozen statistics to a checkpoint file.

        For pretrained backends this stores identity metadata and stats
        but NOT the large pretrained weights (they are reloaded from source).
        For scratch backends this stores the full weight dict.
        """
        raise NotImplementedError

    # ── Normalization helpers (concrete, shared by both backends) ─────────────
    def normalise(self, scaled_latents: torch.Tensor) -> torch.Tensor:
        """
        Map native latents → approximately N(0,1) using frozen training-set stats.

        Parameters
        ----------
        scaled_latents : (B, 4, H', W') float32

        Returns
        -------
        (B, 4, H', W') float32 — normalised latents (unbounded, never clamped)
        """
        return (scaled_latents - self.latent_mean) / self.latent_std

    def denormalise(self, normalised_latents: torch.Tensor) -> torch.Tensor:
        """
        Inverse of normalise().

        Parameters
        ----------
        normalised_latents : (B, 4, H', W') float32

        Returns
        -------
        (B, 4, H', W') float32 — native scaled latents
        """
        return normalised_latents * self.latent_std + self.latent_mean

    def decode_normalised(self, normalised_latents: torch.Tensor) -> torch.Tensor:
        """
        Convenience: denormalise then decode in one call.

        Parameters
        ----------
        normalised_latents : (B, 4, H', W') float32

        Returns
        -------
        (B, 3, H, W) float32 in [-1, 1]
        """
        return self.decode(self.denormalise(normalised_latents))

    # ── Statistics computation ────────────────────────────────────────────────
    def compute_and_freeze_stats(
        self,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
    ) -> None:
        """
        Encode the full training set and compute per-channel mean/std.

        Statistics are frozen immediately after computation; calling this
        a second time raises RuntimeError.

        Parameters
        ----------
        dataloader : yields (images, _) batches; images are (B, 3, H, W) in [-1, 1]
        device     : device to run encoding on
        """
        if self._stats_frozen:
            raise RuntimeError(
                "compute_and_freeze_stats() has already been called on this codec. "
                "Recomputing would silently change the normalization of any cached "
                "latents that used the previous stats."
            )

        all_latents = []
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    images = batch[0]
                else:
                    images = batch
                images = images.to(device)
                z = self.encode_mean(images)  # (B, 4, 16, 16)
                all_latents.append(z.cpu())

        latents = torch.cat(all_latents, dim=0)  # (N, 4, 16, 16)
        # Compute per-channel mean and std over all batch and spatial axes.
        # latents has shape (N, C, H, W) → reduce over dims 0, 2, 3.
        mean = latents.mean(dim=(0, 2, 3), keepdim=True)   # (1, C, 1, 1)
        std = latents.std(dim=(0, 2, 3), keepdim=True)     # (1, C, 1, 1)

        # Guard: std must be strictly positive for normalization to be invertible.
        if not (std > 0).all():
            raise RuntimeError(
                f"Computed latent_std has non-positive channels: {std.flatten().tolist()}. "
                f"This indicates a degenerate codec or empty dataset."
            )

        # Reshape to (1, C, 1, 1) to match the required checkpoint shape.
        self._latent_mean = mean.reshape(1, REQUIRED_LATENT_CHANNELS, 1, 1)
        self._latent_std = std.reshape(1, REQUIRED_LATENT_CHANNELS, 1, 1)
        self._stats_frozen = True

    # ── Helpers for subclasses ────────────────────────────────────────────────
    @staticmethod
    def _sha256_of_state_dict(state_dict: dict) -> str:
        """
        Produce a stable hex SHA-256 digest of a PyTorch state dict.

        Tensors are sorted by key name, then each is converted to bytes via
        numpy for a deterministic representation that does not depend on
        storage order or metadata.
        """
        h = hashlib.sha256()
        for key in sorted(state_dict.keys()):
            tensor = state_dict[key]
            # Ensure CPU, contiguous, float32 for cross-platform stability.
            h.update(key.encode("utf-8"))
            h.update(tensor.cpu().contiguous().float().numpy().tobytes())
        return h.hexdigest()

    @staticmethod
    def _stats_to_list(t: torch.Tensor) -> list:
        """Serialise a (1,4,1,1) tensor to a nested Python list for JSON/torch.save."""
        return t.cpu().tolist()

    @staticmethod
    def _stats_from_list(raw: list, device: torch.device) -> torch.Tensor:
        """Deserialise a nested list to a (1,4,1,1) float32 tensor on device."""
        return torch.tensor(raw, dtype=torch.float32, device=device)

    def _load_frozen_stats(
        self,
        latent_mean_raw: list,
        latent_std_raw: list,
        device: torch.device,
    ) -> None:
        """
        Restore pre-computed statistics from a checkpoint without re-encoding.
        Called by subclass constructors when loading a frozen codec checkpoint.
        """
        mean = self._stats_from_list(latent_mean_raw, device)
        std = self._stats_from_list(latent_std_raw, device)
        if mean.shape != (1, REQUIRED_LATENT_CHANNELS, 1, 1):
            raise CodecCheckpointError(
                f"latent_mean shape mismatch: got {tuple(mean.shape)}"
            )
        if std.shape != (1, REQUIRED_LATENT_CHANNELS, 1, 1):
            raise CodecCheckpointError(
                f"latent_std shape mismatch: got {tuple(std.shape)}"
            )
        if not (std > 0).all():
            raise CodecCheckpointError(
                f"Loaded latent_std has non-positive channels: {std.flatten().tolist()}"
            )
        self._latent_mean = mean
        self._latent_std = std
        self._stats_frozen = True

    def _assert_pixel_input(self, images: torch.Tensor) -> None:
        """Validate pixel tensor shape before encoding."""
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(
                f"Expected pixel input of shape (B, 3, H, W), got {tuple(images.shape)}"
            )

    def _assert_latent_input(self, z: torch.Tensor, *, name: str = "latent") -> None:
        """Validate latent tensor channel count before decoding."""
        if z.ndim != 4 or z.shape[1] != REQUIRED_LATENT_CHANNELS:
            raise ValueError(
                f"Expected {name} of shape (B, {REQUIRED_LATENT_CHANNELS}, H', W'), "
                f"got {tuple(z.shape)}"
            )
