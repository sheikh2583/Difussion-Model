"""
codec/cache_latents.py — Content-addressed latent caching.

OPERATOR USE ONLY — do not execute during agent implementation.

Encodes CelebA with a validated codec and caches normalised latents to disk.
Works with both PretrainedKLVAE and ScratchKLVAE through load_codec().

Cache identity
──────────────
The cache directory name is derived from a content hash that covers:
  • codec_weights_sha256  (encoder/decoder identity)
  • codec_source_revision
  • split                 ("train" | "valid")
  • posterior_mode        ("quantized" or "continuous" for VQ, "mean" for KL)
  • normalization_schema  ("v1" — mean-std normalisation with frozen training stats)
  • preprocessing_schema  ("celeba_center_crop_178_resize_64_norm_m1p1")

This means multiple codec caches can coexist.  The dataset (Codex-owned
CelebALatentDataset) matches by reading cache manifest files and selecting
the one whose content_hash matches the codec checkpoint in the config.

Directory layout
────────────────
  <output_dir>/
    latents_<hash>/               ← final atomic publish
      latents_train.pt            ← (N, 4, H, W) float32 normalised
      latents_valid.pt
      manifest.json               ← full metadata + SHA-256 hashes

Atomicity
─────────
Written to a staging directory first, then renamed into place.  A partial
or interrupted cache is never visible at the final path.

Operator usage
──────────────
Prerequisites:
  • A validated codec checkpoint from validate_codec.py
  • CelebA available at --celeba-root

Run:
    python codec/cache_latents.py \\
        --codec-path ./results/codecs/accepted_codec.pt \\
        --celeba-root ./data/raw \\
        --output-dir ./data/latent_cache \\
        --split both \\
        --batch-size 64 \\
        --device cuda

The manifest.json contains the content_hash that CelebALatentDataset uses
to verify it is consuming exactly the right cache.
"""
import argparse
import hashlib
import json
import os
import sys
import time


_PREPROCESSING_SCHEMA = "celeba_center_crop_178_resize_64_norm_m1p1"
_NORMALIZATION_SCHEMA = "v1_mean_std_frozen_train"


def _main() -> int:
    import torch

    args = _parse_args()
    device = torch.device(args.device)

    print(f"\n{'='*70}")
    print("  Latent caching — content-addressed")
    print(f"{'='*70}")
    print(f"  Codec path    : {args.codec_path}")
    print(f"  CelebA root   : {args.celeba_root}")
    print(f"  Output dir    : {args.output_dir}")
    print(f"  Split(s)      : {args.split}")
    print(f"  Device        : {device}")
    print()

    # ── Load codec ──────────────────────────────────────────────────────────
    print("[1/3] Loading codec …")
    from codec.codec_factory import load_codec
    source_codec_path = os.path.abspath(args.codec_path)
    args.source_codec_path = source_codec_path
    if args.continuous:
        output_codec_path = args.continuous_codec_output or _default_continuous_codec_path(
            source_codec_path
        )
        output_codec_path = os.path.abspath(output_codec_path)
        if output_codec_path == source_codec_path:
            raise ValueError("Continuous mode requires a separate codec checkpoint path")

        source_codec = load_codec(
            source_codec_path, device, require_frozen=False, continuous=True
        )
        if os.path.isfile(output_codec_path):
            codec = load_codec(output_codec_path, device, require_frozen=True)
            if codec.posterior_mode != "continuous":
                raise ValueError(
                    f"Continuous codec output has posterior_mode={codec.posterior_mode!r}: "
                    f"{output_codec_path}"
                )
            if codec._codec_weights_sha256 != source_codec._codec_weights_sha256:
                raise ValueError(
                    "Existing continuous codec checkpoint was built from different "
                    "weights; choose another --continuous-codec-output path."
                )
        else:
            # Continuous mode has a different latent distribution. Never reuse
            # quantized statistics; compute them from the training split only.
            from config.config import DatasetConfig
            from data.celeba import get_dataloaders

            stats_cfg = DatasetConfig(
                name="celeba", root=args.celeba_root,
                image_size=64, num_workers=args.num_workers
            )
            train_loader, _ = get_dataloaders(
                stats_cfg, args.batch_size, seed=0
            )
            source_codec.compute_and_freeze_stats(train_loader, device)
            codec = source_codec
            codec.save(output_codec_path)
            print(f"    Saved continuous codec checkpoint: {output_codec_path}")
        args.codec_path = output_codec_path
    else:
        codec = load_codec(source_codec_path, device, require_frozen=True)
    print(f"    ✓ codec_type = {type(codec).__name__}")

    # Derive content hash for cache identity
    content_hash = _build_content_hash(codec, args.split)
    print(f"    ✓ Cache content hash : {content_hash[:16]}…")

    # Check for existing cache
    cache_dir = os.path.join(args.output_dir, f"latents_{content_hash[:16]}")
    manifest_path = os.path.join(cache_dir, "manifest.json")
    if os.path.isdir(cache_dir) and os.path.isfile(manifest_path):
        print(f"\n  Cache already exists at: {cache_dir}")
        _verify_existing_cache(cache_dir, manifest_path, content_hash)
        print("  ✓ Existing cache verified — nothing to do.\n")
        return 0

    # ── Load CelebA ─────────────────────────────────────────────────────────
    print("[2/3] Loading CelebA dataset …")
    from config.config import DatasetConfig
    from data.celeba import get_dataloaders
    ds_cfg = DatasetConfig(
        name="celeba", root=args.celeba_root,
        image_size=64, num_workers=args.num_workers
    )
    train_loader, val_loader = get_dataloaders(ds_cfg, args.batch_size, seed=0)

    splits_to_encode = []
    if args.split in ("train", "both"):
        splits_to_encode.append(("train", train_loader))
    if args.split in ("valid", "both"):
        splits_to_encode.append(("valid", val_loader))

    # ── Encode into staging directory ───────────────────────────────────────
    print("[3/3] Encoding latents …")
    staging_dir = cache_dir + f".staging-{os.getpid()}"
    os.makedirs(staging_dir, exist_ok=True)

    try:
        file_hashes = {}
        for split_name, loader in splits_to_encode:
            print(f"    Encoding {split_name} split …")
            t0 = time.time()
            latents = _encode_split(codec, loader, device)
            elapsed = time.time() - t0
            print(f"      Shape: {tuple(latents.shape)}  |  Time: {elapsed:.1f}s")

            fname = f"latents_{split_name}.pt"
            fpath = os.path.join(staging_dir, fname)
            torch.save(latents, fpath)
            file_hashes[fname] = _sha256_file(fpath)
            print(f"      SHA-256: {file_hashes[fname][:16]}…")

        # Write manifest
        manifest = _build_manifest(codec, args, content_hash, file_hashes, splits_to_encode)
        manifest_staging = os.path.join(staging_dir, "manifest.json")
        with open(manifest_staging, "w") as f:
            json.dump(manifest, f, indent=2)

        # Atomic publish: rename staging → final
        os.rename(staging_dir, cache_dir)
        print(f"\n  ✓ Cache published → {cache_dir}")

    except Exception:
        # Clean up staging directory on failure
        import shutil
        if os.path.exists(staging_dir):
            shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    print(f"\n{'='*70}")
    print(f"  Latent cache ready: {cache_dir}")
    print(f"  Manifest: {os.path.join(cache_dir, 'manifest.json')}")
    print(f"{'='*70}\n")
    return 0


def _encode_split(codec, loader, device) -> "torch.Tensor":
    """Encode all images and return normalized float32 latents."""
    import torch
    all_latents = []
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch
            images = images.to(device)
            if codec.posterior_mode == "continuous":
                z = codec.encode_continuous(images)
            else:
                z = codec.encode_mean(images)
            z_norm = codec.normalise(z)             # normalised
            all_latents.append(z_norm.cpu().float())
    return torch.cat(all_latents, dim=0)


def _build_content_hash(codec, split: str) -> str:
    """
    Build a stable cache identity hash from codec identity + pipeline parameters.

    Hash inputs (all must be present for cache identity):
      • codec_weights_sha256     — encoder/decoder weight identity
      • codec_source_revision    — source commit/tag
      • split                    — "train", "valid", or "both"
      • posterior_mode           — codec-defined deterministic policy
      • normalization_schema     — "v1_mean_std_frozen_train"
      • preprocessing_schema     — "celeba_center_crop_178_resize_64_norm_m1p1"
      • latent_channels          — codec-defined (3 for primary VQ-f4)
      • spatial_factor           — codec-defined (4 for primary VQ-f4)
      • pixel_size               — 64

    Each field is hashed with a separator to prevent cross-field collisions.
    """
    h = hashlib.sha256()

    def _feed(label: str, value: str) -> None:
        # Label + NUL + value + NUL ensures no two fields can collide.
        h.update(label.encode("utf-8") + b"\x00" + value.encode("utf-8") + b"\x00")

    _feed("weights_sha256",        getattr(codec, "_codec_weights_sha256", "unknown"))
    _feed("source_revision",       getattr(codec, "_codec_source_revision", "unknown"))
    _feed("split",                 split)
    _feed("posterior_mode",        codec.posterior_mode)
    _feed("normalization_schema",  _NORMALIZATION_SCHEMA)
    _feed("preprocessing_schema",  _PREPROCESSING_SCHEMA)
    _feed("latent_channels",       str(codec.latent_channels))
    _feed("spatial_factor",        str(codec.spatial_factor))
    _feed("pixel_size",            str(codec.pixel_size))

    return h.hexdigest()



def _sha256_file(path: str) -> str:
    """SHA-256 of the complete file at path."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_manifest(codec, args, content_hash, file_hashes, splits_encoded) -> dict:
    """Build the manifest dict stored alongside the cached latents."""
    import datetime

    codec_src = getattr(codec, "_codec_source", "unknown")
    codec_rev = getattr(codec, "_codec_source_revision", "unknown")
    codec_sha = getattr(codec, "_codec_weights_sha256", "unknown")

    return {
        "schema_version": 1,
        "content_hash": content_hash,
        "codec_type": type(codec).__name__,
        "codec_source": codec_src,
        "codec_source_revision": codec_rev,
        "codec_weights_sha256": codec_sha,
        "codec_checkpoint_path": os.path.abspath(args.codec_path),
        "source_codec_checkpoint_path": os.path.abspath(
            getattr(args, "source_codec_path", args.codec_path)
        ),
        "posterior_mode": codec.posterior_mode,
        "normalization_schema": _NORMALIZATION_SCHEMA,
        "preprocessing_schema": _PREPROCESSING_SCHEMA,
        "latent_channels": codec.latent_channels,
        "spatial_factor": codec.spatial_factor,
        "pixel_size": codec.pixel_size,
        "latent_mean": codec._stats_to_list(codec.latent_mean),
        "latent_std": codec._stats_to_list(codec.latent_std),
        "splits": [s for s, _ in splits_encoded],
        "file_hashes": file_hashes,
        "dataset": "celeba",
        "celeba_root": os.path.abspath(args.celeba_root),
        "batch_size_used": args.batch_size,
        "created_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    }


def _verify_existing_cache(cache_dir: str, manifest_path: str, expected_hash: str) -> None:
    """Verify that an existing cache matches the expected content hash and file hashes."""
    import json

    with open(manifest_path) as f:
        manifest = json.load(f)

    if manifest.get("content_hash") != expected_hash:
        raise RuntimeError(
            f"Content hash mismatch in existing cache at '{cache_dir}'.\n"
            f"  Expected : {expected_hash}\n"
            f"  Found    : {manifest.get('content_hash')}\n"
            f"Delete the cache directory and re-run to rebuild."
        )

    for fname, expected_sha in manifest.get("file_hashes", {}).items():
        fpath = os.path.join(cache_dir, fname)
        if not os.path.isfile(fpath):
            raise RuntimeError(
                f"Cache manifest references '{fpath}' but the file is missing. "
                f"Delete the cache directory and re-run."
            )
        actual_sha = _sha256_file(fpath)
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"File hash mismatch for '{fpath}'.\n"
                f"  Expected : {expected_sha}\n"
                f"  Actual   : {actual_sha}\n"
                f"The cache file has been corrupted.  Delete and re-run."
            )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Encode CelebA with a validated codec and cache normalised latents."
    )
    p.add_argument("--codec-path", required=True,
                   help="Path to accepted_codec.pt from validate_codec.py")
    p.add_argument("--celeba-root", default="./data/raw")
    p.add_argument("--output-dir", default="./data/latent_cache")
    p.add_argument("--split", choices=["train", "valid", "both"], default="both")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--continuous", action="store_true",
        help=(
            "Bypass VQ quantization and cache pre-quantization continuous "
            "encoder output.  Produces a separate content-addressed cache "
            "that must be used with a continuous-mode codec.  Required for "
            "FM-compatible latent training."
        ),
    )
    p.add_argument(
        "--continuous-codec-output",
        default=None,
        help=(
            "Path for the separate continuous-mode checkpoint containing "
            "continuous training-set normalization statistics. Defaults to "
            "<source codec stem>_continuous.pt."
        ),
    )
    return p.parse_args()


def _default_continuous_codec_path(source_path: str) -> str:
    stem, extension = os.path.splitext(source_path)
    if not extension:
        raise ValueError(
            f"Cannot derive continuous checkpoint name from source path {source_path!r}; "
            "pass --continuous-codec-output explicitly."
        )
    return f"{stem}_continuous{extension}"


if __name__ == "__main__":
    sys.exit(_main())
