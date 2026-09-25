"""
codec/codec_factory.py — Backend-neutral codec loader.

load_codec(path, device) is the single dispatch point used by all pipeline
code (cache_latents, evaluator, smoke_latent).  It reads the codec_type from
the checkpoint and constructs the correct backend without the caller needing
to know which class to import.

Lazy imports
────────────
Both backends are imported only when actually needed so:
  - pretrained_vae.py can be absent or have missing deps (diffusers) and
    only the pretrained path will fail, not the scratch path and vice-versa.
  - Neither backend file is imported during normal pixel-only training.

Validation
──────────
After construction the factory validates the metadata again through
validate_checkpoint_metadata() to guarantee that any codec returned by
load_codec() satisfies the full schema contract.  Backend constructors
may also validate internally, but the factory provides a uniform guarantee.
"""
import os
from typing import Optional

import torch

from codec.base import (
    CodecCheckpointError,
    validate_checkpoint_metadata,
)

_SUPPORTED_CODEC_TYPES = frozenset({"pretrained_vq_f4", "scratch_kl_vae"})


def load_codec(
    path: str,
    device: torch.device,
    require_frozen: bool = True,
    *,
    continuous: bool | None = None,
) -> "codec.base.BaseCodec":  # type: ignore[name-defined]
    """
    Load a codec checkpoint and return a fully-initialised BaseCodec instance.

    The function reads the minimal metadata needed for dispatch, validates the
    full schema, constructs the correct backend, and returns it ready to use.

    Parameters
    ----------
    path            : path to a .pt codec checkpoint written by BaseCodec.save()
    device          : device to place the codec on
    require_frozen  : if True (default), reject checkpoints without frozen stats
    continuous      : override the pretrained VQ posterior mode. If this
                      differs from checkpoint metadata, load without frozen
                      statistics; callers must compute mode-specific stats.

    Returns
    -------
    A pretrained or scratch codec satisfying BaseCodec.

    Raises
    ------
    FileNotFoundError      if path does not exist
    CodecCheckpointError   if the checkpoint fails any validation step
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Codec checkpoint not found: '{path}'\n"
            f"Run 'python codec/validate_codec.py' to create it."
        )

    # Load the full checkpoint to extract metadata for dispatch.
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise CodecCheckpointError(
            f"Failed to load codec checkpoint '{path}': {exc}"
        ) from exc

    if not isinstance(ckpt, dict) or "metadata" not in ckpt:
        raise CodecCheckpointError(
            f"Codec checkpoint '{path}' does not contain a 'metadata' key. "
            f"This file was not produced by BaseCodec.save()."
        )

    meta = ckpt["metadata"]

    # Validate the full schema before dispatch — this is the authoritative check.
    validate_checkpoint_metadata(meta, path=path, require_frozen=require_frozen)

    codec_type = meta["codec_type"]
    if codec_type not in _SUPPORTED_CODEC_TYPES:
        raise CodecCheckpointError(
            f"Unknown codec_type={codec_type!r} in checkpoint '{path}'. "
            f"Supported types: {sorted(_SUPPORTED_CODEC_TYPES)}"
        )

    # ── Dispatch to the correct backend ──────────────────────────────────────
    if codec_type == "pretrained_vq_f4":
        try:
            from codec.pretrained_vae import PretrainedVQCodec
        except ImportError as exc:
            raise ImportError(
                f"Cannot load pretrained_vq_f4 codec from '{path}': {exc}"
            ) from exc
        load_kwargs = {"require_frozen": require_frozen}
        if continuous is not None:
            load_kwargs["continuous"] = continuous
        return PretrainedVQCodec.from_checkpoint(path, device, **load_kwargs)

    if codec_type == "scratch_kl_vae":
        if continuous:
            raise CodecCheckpointError(
                "continuous mode is supported only by pretrained_vq_f4 codecs"
            )
        # Lazy import; scratch_vae.py is owned by Codex.
        try:
            from codec.scratch_vae import ScratchKLVAE  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                f"Cannot load scratch_kl_vae codec from '{path}': {exc}\n"
                f"Ensure codec/scratch_vae.py (Codex-owned) is present."
            ) from exc
        return ScratchKLVAE.from_checkpoint(  # type: ignore[attr-defined]
            path, device, require_frozen=require_frozen
        )

    # Should be unreachable given the check above, but be defensive.
    raise CodecCheckpointError(f"Unhandled codec_type={codec_type!r}")
