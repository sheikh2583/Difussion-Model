"""
Common evaluation pipeline. Does not need to know which algorithm
produced the images — it only consumes algorithm.sample() output via
the Sampler, plus the cached real-data reference statistics.

Latent extension (2026-09-21)
─────────────────────────────
When cfg.dataset.name == "celeba_latent" the evaluator:

  1. Uses a SEPARATE FID reference cache (latent experiment cache filename).
     The reference is always built from RAW CelebA validation pixels — never
     from cached latents or reconstructions.

  2. Decodes generated normalised latent tensors to (B,3,64,64) in bounded
     batches before feeding them to FID/IS.  The decoder is loaded lazily
     from the latent cache manifest once and reused for the whole evaluation.

  3. Records backbone sampling time and decoder time SEPARATELY in
     ResultRecord.backbone_sampling_time / ResultRecord.decoder_time.

  4. Never clamps latent tensors to [-1, 1].  The clamp is applied only
     to decoded pixel images before FID/IS, consistently with metrics.py.

Pixel path (cifar10, celeba) is unchanged in behaviour.
"""
import json
import os
import re
from typing import List, Optional

import torch
from torch.utils.data import DataLoader

from config.config import ExperimentConfig
from evaluation.metrics import (
    cache_real_images,
    compute_fid_from_prepared,
    compute_inception_score,
    load_real_images,
    prepare_fid,
)
from sampling.sampler import Sampler
from utils.results import ResultRecord, ResultsWriter
from utils.timing import timer, peak_gpu_memory_mb, reset_peak_gpu_memory
from utils.plots import (
    plot_fid_vs_nfe,
    plot_fid_vs_sampling_time,
    plot_gpu_memory_comparison,
    plot_is_vs_nfe,
    plot_loss_vs_epoch,
    plot_sampling_time_vs_nfe,
    plot_training_time_comparison,
)

# ── Latent experiment detection ───────────────────────────────────────────────
_LATENT_DATASET_NAME = "celeba_latent"

# Decode generated latents in this many samples per VAE call.
# Prevents holding all 5,000 decoder activations simultaneously.
_DEFAULT_DECODE_BATCH = 32


def _fid_reference_meta_path(cache_path: str) -> str:
    return cache_path + ".meta.json"


# ── Pixel-space FID reference (original behaviour, unchanged) ─────────────────
def ensure_fid_reference(cfg: ExperimentConfig, real_loader: DataLoader,
                          device: torch.device) -> str:
    """
    Computes (once) and caches FID reference statistics from real data,
    so every algorithm's evaluation reuses the exact same reference —
    never recomputed per run.

    A small metadata file is cached alongside the stats (image count,
    resolution, channels) and checked on every reuse. If a later run
    requests a reference set with different settings but happens to
    point at the same cache path, this raises explicitly rather than
    silently comparing FID against a mismatched reference distribution.
    """
    cache_path = cfg.evaluation.fid_reference_cache
    meta_path = _fid_reference_meta_path(cache_path)

    expected_meta = {
        "num_images": cfg.evaluation.num_generated_samples,
        "image_size": cfg.dataset.image_size,
        "channels": cfg.backbone.in_channels,
    }

    if os.path.exists(cache_path):
        if not os.path.exists(meta_path):
            raise RuntimeError(
                f"FID reference cache '{cache_path}' exists but its metadata "
                f"file '{meta_path}' is missing, so it cannot be validated as "
                f"matching the current config. Delete the stale cache file "
                f"and re-run, or restore its metadata file."
            )
        with open(meta_path, "r") as f:
            cached_meta = json.load(f)
        if cached_meta != expected_meta:
            raise RuntimeError(
                f"FID reference cache '{cache_path}' was built with "
                f"{cached_meta} but the current run requests {expected_meta}. "
                f"Reusing it would silently bias the FID comparison. Use a "
                f"different `evaluation.fid_reference_cache` path per "
                f"resolution/sample-count setting, or delete the stale cache."
            )
        return cache_path

    real_batches = []
    n_needed = cfg.evaluation.num_generated_samples
    n_collected = 0
    for images, _ in real_loader:
        real_batches.append(images)
        n_collected += images.shape[0]
        if n_collected >= n_needed:
            break

    if n_collected < n_needed:
        raise RuntimeError(
            f"Requested {n_needed} real reference images for FID, but the "
            f"dataset only provided {n_collected}. Reduce "
            f"evaluation.num_generated_samples or provide a larger reference "
            f"split rather than silently evaluating against fewer images."
        )

    real_images = torch.cat(real_batches, dim=0)[:n_needed]

    cache_real_images(real_images, cache_path)
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(expected_meta, f, indent=2)
    return cache_path


# ── Latent-experiment FID reference (separate function, separate cache path) ──
# Module-level alias so tests can mock it via mock.patch("evaluation.evaluator._celeba_get_dataloaders")
from data.celeba import get_dataloaders as _celeba_get_dataloaders


def ensure_fid_reference_latent(
    cfg: ExperimentConfig,
    device: torch.device,
    n_images: int = 5000,
) -> str:
    """
    Build (once) a FID reference from RAW CelebA validation pixels for use with
    latent experiments.

    This function always loads raw CelebA pixels — NEVER cached latents or
    reconstructions.  The FID reference for latent experiments describes
    decoded output in the 64×64 RGB space, exactly like the pixel experiments,
    but uses a separate cache path to avoid collisions.

    The reference metadata records:
        num_images   = n_images (5000)
        image_size   = 64
        channels     = 3  (RGB decoded output)

    Parameters
    ----------
    cfg      : ExperimentConfig — used for fid_reference_cache path and CelebA root
    device   : evaluation device
    n_images : number of real images to use (must be 5000 per project contract)

    Returns
    -------
    Path to the FID reference .npz / .pt cache.
    """
    cache_path = cfg.evaluation.fid_reference_cache
    meta_path = _fid_reference_meta_path(cache_path)

    # For latent experiments the FID reference always describes RGB decoded images.
    expected_meta = {
        "num_images": n_images,
        "image_size": 64,       # decoded pixel size, always 64
        "channels": 3,          # RGB
        "source": "raw_celeba_validation_pixels",  # explicit provenance
    }

    if os.path.exists(cache_path):
        if not os.path.exists(meta_path):
            raise RuntimeError(
                f"FID reference cache '{cache_path}' exists but its metadata "
                f"'{meta_path}' is missing. Delete the stale cache and re-run."
            )
        with open(meta_path, "r") as f:
            cached_meta = json.load(f)
        if cached_meta != expected_meta:
            raise RuntimeError(
                f"FID reference cache '{cache_path}' was built with "
                f"{cached_meta} but the current run expects {expected_meta}. "
                f"Use a different fid_reference_cache path for latent experiments "
                f"or delete the stale cache."
            )
        return cache_path

    # Import raw CelebA loader (not latent dataset — always raw pixels).
    # Uses module-level _celeba_get_dataloaders alias so tests can patch it.
    from config.config import DatasetConfig
    ds_cfg = DatasetConfig(
        name="celeba",
        root=cfg.dataset.root,
        image_size=64,
        num_workers=getattr(cfg.dataset, "num_workers", 4),
    )
    # Seed=0 for reproducibility; we only read the val split.
    _, val_loader = _celeba_get_dataloaders(ds_cfg, batch_size=64, seed=0)


    real_batches = []
    n_collected = 0
    for images, _ in val_loader:
        real_batches.append(images)
        n_collected += images.shape[0]
        if n_collected >= n_images:
            break

    if n_collected < n_images:
        raise RuntimeError(
            f"Requested {n_images} real reference images for latent FID, "
            f"but CelebA validation only provided {n_collected}. "
            f"Reduce n_images or use a larger validation split."
        )

    real_images = torch.cat(real_batches, dim=0)[:n_images]
    cache_real_images(real_images, cache_path)
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(expected_meta, f, indent=2)
    return cache_path


# ── Codec loader for evaluator (lazy, single instance per Evaluator) ──────────
def _load_codec_from_cache_manifest(latent_cache_dir: str, device: torch.device):
    """
    Read the latent cache manifest to find the codec checkpoint path,
    then load the codec via the factory.

    Parameters
    ----------
    latent_cache_dir : directory written by cache_latents.py
    device           : target device for the codec

    Returns
    -------
    A BaseCodec instance loaded with frozen stats.
    """
    manifest_path = os.path.join(latent_cache_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        raise RuntimeError(
            f"Latent cache manifest not found at '{manifest_path}'. "
            f"Run 'python codec/cache_latents.py' to create the cache."
        )
    with open(manifest_path) as f:
        manifest = json.load(f)

    codec_path = manifest.get("codec_checkpoint_path")
    if not codec_path:
        raise RuntimeError(
            f"Latent cache manifest '{manifest_path}' does not contain "
            f"'codec_checkpoint_path'.  Rebuild the cache with the current "
            f"version of cache_latents.py."
        )
    if not os.path.isfile(codec_path):
        raise RuntimeError(
            f"Codec checkpoint referenced by cache manifest not found: '{codec_path}'. "
            f"Ensure the codec checkpoint is at the recorded path."
        )

    try:
        from codec.codec_factory import load_codec
    except ImportError as exc:
        raise ImportError(
            f"Cannot import codec factory for latent evaluation: {exc}. "
            f"Install the codec dependencies (diffusers>=0.21) or use a pixel config."
        ) from exc

    return load_codec(codec_path, device, require_frozen=True)


def _load_codec_checkpoint(codec_checkpoint: str, device: torch.device):
    """Load the explicitly configured codec checkpoint for latent decoding."""
    if not codec_checkpoint:
        raise RuntimeError(
            "Latent evaluation requires dataset.codec_checkpoint. "
            "Set it to the accepted codec checkpoint used to build the cache."
        )
    if not os.path.isfile(codec_checkpoint):
        raise RuntimeError(
            f"Configured codec checkpoint not found: '{codec_checkpoint}'."
        )
    from codec.codec_factory import load_codec
    return load_codec(codec_checkpoint, device, require_frozen=True)


def _decode_latents_batched(
    codec,
    latents: torch.Tensor,
    decode_batch_size: int = _DEFAULT_DECODE_BATCH,
) -> torch.Tensor:
    """
    Decode a batch of normalised latents to pixels in bounded sub-batches.

    Parameters
    ----------
    codec            : BaseCodec with frozen stats
    latents          : (N, 4, H', W') float32 normalised latents on CPU
    decode_batch_size: number of latents decoded per VAE call

    Returns
    -------
    (N, 3, H, W) float32 in [-1, 1] on CPU
    """
    decoded_chunks = []
    if hasattr(codec, "_vae"):
        device = next(iter(codec._vae.parameters())).device
    elif isinstance(codec, torch.nn.Module):
        device = next(codec.parameters()).device
    else:
        device = codec.latent_mean.device
    with torch.no_grad():
        for start in range(0, latents.shape[0], decode_batch_size):
            chunk = latents[start: start + decode_batch_size].to(device)
            decoded_chunks.append(codec.decode_normalised(chunk).cpu())
    return torch.cat(decoded_chunks, dim=0)


# ── Main Evaluator class ──────────────────────────────────────────────────────
class Evaluator:
    def __init__(self, cfg: ExperimentConfig, run_dir: str, device: torch.device):
        self.cfg = cfg
        self.run_dir = run_dir
        self.device = device
        run_name = os.path.basename(os.path.normpath(run_dir))
        self.results = ResultsWriter(run_dir, run_name)

        # Lazily-loaded codec for latent experiments (None for pixel configs).
        self._codec = None

    def _is_latent_experiment(self) -> bool:
        return self.cfg.dataset.name == _LATENT_DATASET_NAME

    def _get_codec(self):
        """Load the codec once and cache it for the lifetime of this Evaluator."""
        if self._codec is None:
            self._codec = _load_codec_checkpoint(
                getattr(self.cfg.dataset, "codec_checkpoint", ""), self.device
            )
        return self._codec

    def evaluate(self, sampler: Sampler, nfe_values: List[int],
                 make_plots: bool = False,
                 checkpoint_path: str = None,
                 current_epoch: int = None,
                 decode_batch_size: int = _DEFAULT_DECODE_BATCH) -> None:
        """
        Run evaluation for all requested NFE values.

        For latent experiments (dataset.name == "celeba_latent"):
          - generated tensors are normalised latents → decoded before FID/IS
          - backbone_sampling_time and decoder_time are recorded separately

        For pixel experiments (cifar10, celeba):
          - behaviour is identical to the original implementation
          - backbone_sampling_time = sampling_time (decoder_time = None)
        """
        if current_epoch is None and checkpoint_path:
            match = re.search(r"epoch(\d+)", str(checkpoint_path))
            current_epoch = int(match.group(1)) if match else None

        is_latent = self._is_latent_experiment()

        real_images = None
        fid_metric = None
        if "fid" in self.cfg.evaluation.metrics:
            real_images = load_real_images(
                self.cfg.evaluation.fid_reference_cache, torch.device("cpu"))
            fid_metric = prepare_fid(real_images, self.device)

        for nfe in nfe_values:
            sample_count = self.cfg.evaluation.num_generated_samples
            reset_peak_gpu_memory(self.device)

            # ── Generate samples (backbone only — time this separately) ──────
            with timer(self.device) as sample_timer:
                raw_output = sampler.generate_for_evaluation(sample_count, nfe)
            backbone_elapsed = sample_timer["elapsed"]

            # ── Decode latents if this is a latent experiment ─────────────
            decoder_elapsed: Optional[float] = None
            if is_latent:
                codec = self._get_codec()
                with timer(self.device) as decode_timer:
                    images = _decode_latents_batched(
                        codec, raw_output, decode_batch_size
                    )
                decoder_elapsed = decode_timer["elapsed"]
                # Total sampling time = backbone + decoder (for the sampling record)
                total_elapsed = backbone_elapsed + decoder_elapsed
            else:
                images = raw_output
                total_elapsed = backbone_elapsed

            # ── Write timing record ────────────────────────────────────────
            self.results.write(ResultRecord(
                algorithm=sampler.algorithm.name(),
                seed=sampler.seed,
                record_type="sampling",
                epoch=current_epoch,
                nfe=nfe,
                sampling_time=total_elapsed,
                time_per_image=total_elapsed / sample_count,
                images_per_second=sample_count / max(total_elapsed, 1e-8),
                peak_gpu_memory=peak_gpu_memory_mb(self.device),
                backbone_sampling_time=backbone_elapsed,
                decoder_time=decoder_elapsed,
            ))

            # ── FID ────────────────────────────────────────────────────────
            fid_score = None
            if fid_metric is not None:
                fid_score = compute_fid_from_prepared(
                    fid_metric, images, self.device
                )

            # ── IS ─────────────────────────────────────────────────────────
            is_mean = is_std = None
            if "is" in self.cfg.evaluation.metrics:
                is_mean, is_std = compute_inception_score(images, self.device)

            self.results.write(ResultRecord(
                algorithm=sampler.algorithm.name(),
                seed=sampler.seed,
                record_type="evaluation",
                epoch=current_epoch,
                nfe=nfe,
                fid=fid_score,
                is_mean=is_mean,
                is_std=is_std,
                checkpoint_path=checkpoint_path,
                num_generated_samples=sample_count,
                backbone_sampling_time=backbone_elapsed,
                decoder_time=decoder_elapsed,
            ))

        if make_plots:
            self._make_plots()

    def _make_plots(self) -> None:
        plot_dir = os.path.join(self.run_dir, "metrics", "plots")
        jsonl_path = self.results.jsonl_path
        plots = (
            (plot_fid_vs_nfe, "fid_vs_nfe.png"),
            (plot_is_vs_nfe, "is_vs_nfe.png"),
            (plot_sampling_time_vs_nfe, "sampling_time_vs_nfe.png"),
            (plot_fid_vs_sampling_time, "fid_vs_sampling_time.png"),
            (plot_loss_vs_epoch, "loss_vs_epoch.png"),
            (plot_training_time_comparison, "training_time.png"),
            (plot_gpu_memory_comparison, "gpu_memory.png"),
        )
        for plot_function, filename in plots:
            plot_function(jsonl_path, os.path.join(plot_dir, filename))
