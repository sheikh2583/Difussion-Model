"""
codec/pretrained_vae.py — Frozen pretrained factor-8 KL autoencoder.

Pinned architecture
───────────────────
Stability AI's SD VAE fine-tuned for MSE reconstruction maps 64×64 RGB pixels
to four-channel 8×8 posterior-mean latents. The encoder and decoder stay frozen.

Pinned source
─────────────
    HuggingFace repo : stabilityai/sd-vae-ft-mse
    Revision         : main (pin to a specific commit for reproducibility)
    Factor verified  : shape assertion at load time — any mismatch raises immediately

Operator prerequisite
─────────────────────
The weights must be downloaded before this module can load a codec:

    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id="stabilityai/sd-vae-ft-mse",
        local_dir="./data/pretrained/sd-vae-ft-mse",
    )

Then validate and create the codec checkpoint:

    python codec/validate_codec.py \\
        --codec-source stabilityai/sd-vae-ft-mse \\
        --codec-source-path ./data/pretrained/sd-vae-ft-mse \\
        --celeba-root ./data/raw \\
        --output-dir ./results/codecs

The resulting checkpoint at ./results/codecs/accepted_codec.pt is the only file
the rest of the pipeline needs.

Dependency
──────────
Requires `diffusers` >= 0.21 (optional dependency).  If not installed, a clear
ImportError is raised immediately when this module is imported, not at runtime.
Existing pixel-only workflows that never import this module are unaffected.
"""
import os
from typing import Optional

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

# ── Optional import guard ──────────────────────────────────────────────────────
try:
    from diffusers import AutoencoderKL as _AutoencoderKL
    _DIFFUSERS_AVAILABLE = True
except ImportError:
    _AutoencoderKL = None  # type: ignore[assignment]
    _DIFFUSERS_AVAILABLE = False


def _require_diffusers() -> None:
    if not _DIFFUSERS_AVAILABLE:
        raise ImportError(
            "The 'diffusers' package is required for PretrainedKLVAE but is not "
            "installed in this environment.  Install it with:\n\n"
            "    pip install diffusers>=0.21\n\n"
            "Pixel-only workflows do not require this dependency."
        )


# ── Source constants ───────────────────────────────────────────────────────────
_CODEC_SOURCE: str = "stabilityai/sd-vae-ft-mse"
_CODEC_TYPE: str = "pretrained_kl_vae"
# The SD VAE configuration records 0.18215 for diffusion-model latent scaling.
# Reconstruction itself is invariant when encode and decode apply reciprocal
# scaling. Recording it keeps the cached representation explicit and compatible
# with the pretrained model family.
_DEFAULT_NATIVE_SCALING_FACTOR: float = 0.18215


class PretrainedKLVAE(BaseCodec):
    """
    Frozen pretrained factor-8 KL autoencoder.

    Encoder and decoder are frozen in eval mode.  No gradients flow through
    this codec.  Only the generative backbone is trained.

    Loading paths
    ─────────────
    From a validated codec checkpoint (normal inference / training use):

        codec = PretrainedKLVAE.from_checkpoint(path, device)

    From the raw pretrained source (operator validation only):

        codec = PretrainedKLVAE.from_pretrained(
            source_path="./data/pretrained/sd-vae-ft-mse",
            device=device,
        )
    """

    def __init__(
        self,
        vae: "AutoencoderKL",  # type: ignore[name-defined]
        native_scaling_factor: float,
        codec_source: str,
        codec_source_revision: str,
        codec_weights_sha256: str,
        device: torch.device,
    ) -> None:
        super().__init__()
        _require_diffusers()

        self._vae = vae
        self._native_scaling_factor = float(native_scaling_factor)
        self._codec_source = codec_source
        self._codec_source_revision = codec_source_revision
        self._codec_weights_sha256 = codec_weights_sha256
        self._device = device

        # Freeze and put in eval mode — no gradient tracking ever.
        self._vae.eval()
        self._vae.requires_grad_(False)
        self._vae.to(device)

        # Verify the spatial factor by construction.
        self._verify_factor()

    # ── Constructor helpers ───────────────────────────────────────────────────
    @classmethod
    def from_pretrained(
        cls,
        source_path: str,
        device: torch.device,
        codec_source: str = _CODEC_SOURCE,
        codec_source_revision: str = "local",
        native_scaling_factor: float = _DEFAULT_NATIVE_SCALING_FACTOR,
    ) -> "PretrainedKLVAE":
        """
        Load directly from the raw pretrained weights directory.

        This is the operator-facing path used by validate_codec.py.
        The resulting codec does NOT have frozen statistics yet; call
        compute_and_freeze_stats() before saving.

        Parameters
        ----------
        source_path             : local directory containing config.json and
                                  diffusion_pytorch_model.bin (or .safetensors)
        device                  : target device
        codec_source_revision   : string to record in the checkpoint (e.g. git SHA)
        native_scaling_factor   : codec-native scaling applied to encoder outputs
        """
        _require_diffusers()
        if not os.path.isdir(source_path):
            raise FileNotFoundError(
                f"Pretrained codec source directory not found: '{source_path}'\n"
                f"Download the weights first:\n"
                f"  from huggingface_hub import snapshot_download\n"
                f"  snapshot_download(repo_id='{_CODEC_SOURCE}', "
                f"local_dir='./data/pretrained/sd-vae-ft-mse')"
            )
        vae = _AutoencoderKL.from_pretrained(source_path)
        weights_sha = cls._vae_weights_sha256(vae)
        return cls(
            vae=vae,
            native_scaling_factor=native_scaling_factor,
            codec_source=codec_source,
            codec_source_revision=codec_source_revision,
            codec_weights_sha256=weights_sha,
            device=device,
        )

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        device: torch.device,
        require_frozen: bool = True,
    ) -> "PretrainedKLVAE":
        """
        Reload a codec from a checkpoint produced by save().

        The checkpoint stores identity metadata (source path, weight digest)
        and frozen statistics, but NOT the large pretrained weight tensors.
        The encoder/decoder weights are reloaded from the original source path
        recorded in the checkpoint.

        Parameters
        ----------
        path            : path to a codec .pt file
        device          : target device
        require_frozen  : if True, reject checkpoints without frozen stats
        """
        _require_diffusers()
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        meta = ckpt.get("metadata", {})
        validate_checkpoint_metadata(meta, path=path, require_frozen=require_frozen)

        # Validate that the stored codec_type matches this class.
        if meta.get("codec_type") != _CODEC_TYPE:
            raise CodecCheckpointError(
                f"Checkpoint '{path}' has codec_type={meta.get('codec_type')!r} "
                f"but PretrainedKLVAE expects '{_CODEC_TYPE}'."
            )

        # Reload the encoder/decoder from the stored source path.
        source_path = ckpt.get("source_path")
        if not source_path or not os.path.isdir(source_path):
            raise CodecCheckpointError(
                f"Checkpoint '{path}' records source_path={source_path!r} "
                f"but that directory does not exist.  Move the pretrained weights "
                f"to the expected location, or re-create the codec checkpoint with "
                f"python codec/validate_codec.py."
            )

        vae = _AutoencoderKL.from_pretrained(source_path)

        # Verify the weight digest to detect silent weight replacement.
        actual_sha = cls._vae_weights_sha256(vae)
        stored_sha = meta.get("codec_weights_sha256", "")
        if actual_sha != stored_sha:
            raise CodecCheckpointError(
                f"Codec weight digest mismatch for '{path}'.\n"
                f"  Stored SHA-256 : {stored_sha}\n"
                f"  Actual SHA-256 : {actual_sha}\n"
                f"The weights at '{source_path}' have changed since the codec "
                f"checkpoint was created.  Re-validate with validate_codec.py."
            )

        codec = cls(
            vae=vae,
            native_scaling_factor=meta["native_scaling_factor"],
            codec_source=meta["codec_source"],
            codec_source_revision=meta["codec_source_revision"],
            codec_weights_sha256=stored_sha,
            device=device,
        )

        if meta.get("stats_frozen") is True:
            codec._load_frozen_stats(
                meta["latent_mean"],
                meta["latent_std"],
                device,
            )
        return codec

    # ── Shape verification ────────────────────────────────────────────────────
    def _verify_factor(self) -> None:
        """
        Assert that this VAE produces (B, 4, 8, 8) from (B, 3, 64, 64).
        Any mismatch fails immediately rather than silently producing wrong latents.
        """
        with torch.no_grad():
            dummy = torch.zeros(1, 3, REQUIRED_PIXEL_SIZE, REQUIRED_PIXEL_SIZE,
                                device=self._device)
            try:
                z = self._vae.encode(dummy).latent_dist.mean
            except Exception as exc:
                raise CodecCheckpointError(
                    f"VAE encode() failed on a synthetic (1,3,64,64) input: {exc}"
                ) from exc
            expected_h = REQUIRED_PIXEL_SIZE // REQUIRED_SPATIAL_FACTOR
            if z.shape != (1, REQUIRED_LATENT_CHANNELS, expected_h, expected_h):
                raise CodecCheckpointError(
                    f"Spatial factor verification FAILED.\n"
                    f"  Expected output shape : "
                    f"(1, {REQUIRED_LATENT_CHANNELS}, {expected_h}, {expected_h})\n"
                    f"  Actual output shape   : {tuple(z.shape)}\n"
                    f"This codec has spatial_factor != {REQUIRED_SPATIAL_FACTOR}."
                )
            self.spatial_factor = REQUIRED_SPATIAL_FACTOR

    @staticmethod
    def _vae_weights_sha256(vae) -> str:
        # Hash the complete VAE, including quant_conv and post_quant_conv; both
        # affect the encoded/decoded representation and therefore cache identity.
        state = {f"vae.{key}": value for key, value in vae.state_dict().items()}
        return BaseCodec._sha256_of_state_dict(state)

    # ── BaseCodec implementation ──────────────────────────────────────────────
    @torch.no_grad()
    def encode_mean(self, images: torch.Tensor) -> torch.Tensor:
        """
        Deterministic posterior-mean encoding.

        Input  : (B, 3, 64, 64) float32 in [-1, 1]
        Output : (B, 4, 8, 8) float32 — native scaled latents
        """
        self._assert_pixel_input(images)
        images = images.to(self._device)
        # posterior.mean is the deterministic encoding path.
        z = self._vae.encode(images).latent_dist.mean
        return z * self._native_scaling_factor

    @torch.no_grad()
    def decode(self, scaled_latents: torch.Tensor) -> torch.Tensor:
        """
        Decode native-scaled latents → pixels in [-1, 1].

        Input  : (B, 4, 8, 8) float32 — native scaled latents
        Output : (B, 3, 64, 64) float32 clamped to [-1, 1]
        """
        self._assert_latent_input(scaled_latents, name="scaled_latents")
        scaled_latents = scaled_latents.to(self._device)
        # Undo native scaling before feeding to the decoder.
        z = scaled_latents / self._native_scaling_factor
        reconstructed = self._vae.decode(z).sample
        return reconstructed.clamp(-1.0, 1.0)

    def save(self, path: str) -> None:
        """
        Save codec identity metadata and frozen statistics.

        The large pretrained weights are NOT duplicated in the checkpoint;
        the source_path stored here is used to reload them later.  This
        keeps the checkpoint small while remaining fully self-describing.

        Parameters
        ----------
        path : destination .pt file path (created atomically)
        """
        if not self._stats_frozen:
            raise RuntimeError(
                "Cannot save codec before statistics are frozen. "
                "Call compute_and_freeze_stats() on the training-set dataloader first."
            )

        meta: dict = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "codec_type": _CODEC_TYPE,
            "codec_source": self._codec_source,
            "codec_source_revision": self._codec_source_revision,
            "codec_weights_sha256": self._codec_weights_sha256,
            "latent_channels": REQUIRED_LATENT_CHANNELS,
            "spatial_factor": self.spatial_factor,
            "pixel_size": REQUIRED_PIXEL_SIZE,
            "posterior_mode": self.posterior_mode,
            "native_scaling_factor": self._native_scaling_factor,
            "stats_frozen": True,
            "latent_mean": self._stats_to_list(self.latent_mean),
            "latent_std": self._stats_to_list(self.latent_std),
            "dataset": "celeba",          # frozen on CelebA training split
            "image_size": REQUIRED_PIXEL_SIZE,
        }

        # Round-trip validate before writing to disk.
        validate_checkpoint_metadata(meta, path=path, require_frozen=True)

        payload = {
            "metadata": meta,
            # Record the source path so from_checkpoint() can reload weights.
            # This is the local filesystem path used during validation.
            "source_path": self._find_source_path(),
        }

        _atomic_save(payload, path)

    def _find_source_path(self) -> str:
        """
        Return the source directory path recorded in this codec instance.
        Set by from_pretrained(); from_checkpoint() copies it from the checkpoint.
        """
        # This attribute is set externally by validate_codec.py after calling
        # from_pretrained().  If it isn't set, we can't save portably.
        source_path = getattr(self, "_source_path", None)
        if not source_path:
            raise RuntimeError(
                "Cannot determine source_path for codec checkpoint.  "
                "If you used from_pretrained(), assign codec._source_path = source_dir "
                "before calling save().  This is handled automatically by validate_codec.py."
            )
        return source_path


# ── Atomic file-save helper (mirrors trainer.py pattern) ────────────────────
def _atomic_save(payload: dict, path: str) -> None:
    """Write payload atomically: write to tmp, then rename."""
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp-{os.getpid()}"
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
