"""
codec/validate_codec.py — Operator-run codec validation and checkpoint creation.

OPERATOR USE ONLY — do not execute during agent implementation.

This script:
  1. Loads a pretrained factor-8 AutoencoderKL from the source weights directory.
  2. Verifies encode/decode shapes and pixel-range for a small validation batch.
  3. Computes frozen per-channel statistics from the CelebA TRAINING set.
  4. Evaluates reconstruction quality on 5,000 CelebA validation images:
       - rFID  < 5      (project gate; see note below)
       - PSNR  > 30 dB
       - min channel std > 0.1 (latent expressiveness gate)
  5. Writes:
       - A codec checkpoint (.pt) containing metadata and frozen stats.
       - A reconstruction grid (.png).
       - A validation report (.json).
  6. Exits nonzero on any failure.

PROJECT GATE NOTE
─────────────────
The rFID < 5 / PSNR > 30 / min-std > 0.1 thresholds are PROJECT-SPECIFIC GATES
chosen to select a codec suitable for this latent diffusion study.  They are NOT
universal literature thresholds and should not be cited as such.  A codec failing
these thresholds under different dataset conditions or resolutions is not
necessarily inadequate for other purposes.

Operator usage
──────────────
Prerequisites:
  1. Download the pretrained codec:
       from huggingface_hub import snapshot_download
       snapshot_download(
           repo_id="stabilityai/sd-vae-ft-mse",
           local_dir="./data/pretrained/sd-vae-ft-mse",
       )
  2. Ensure CelebA is available at --celeba-root (auto-downloads if missing).

Run:
    python codec/validate_codec.py \\
        --codec-source stabilityai/sd-vae-ft-mse \\
        --codec-source-path ./data/pretrained/sd-vae-ft-mse \\
        --celeba-root ./data/raw \\
        --output-dir ./results/codecs \\
        --batch-size 32 \\
        --device cuda

Outputs (on success):
    ./results/codecs/accepted_codec.pt          ← the codec checkpoint
    ./results/codecs/validation_report.json     ← metrics and metadata
    ./results/codecs/reconstruction_grid.png    ← visual quality check
"""
import argparse
import json
import math
import os
import sys
import time

# ── Defer heavy imports so --help works without GPU/CUDA ─────────────────────
def _main() -> int:
    import torch
    import torch.nn.functional as F
    from torchvision.utils import save_image

    args = _parse_args()
    args.codec_source_revision = _resolve_source_revision(
        args.codec_source_path, args.codec_source_revision
    )
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print("  Codec validation — frozen pretrained AutoencoderKL")
    print(f"{'='*70}")
    print(f"  Source path  : {args.codec_source_path}")
    print(f"  CelebA root  : {args.celeba_root}")
    print(f"  Output dir   : {args.output_dir}")
    print(f"  Device       : {device}")
    print(f"  Batch size   : {args.batch_size}")
    print()

    # ── Step 1: Load codec from source weights ──────────────────────────────
    print("[1/6] Loading codec from source weights …")
    from codec.pretrained_vae import PretrainedKLVAE
    codec = PretrainedKLVAE.from_pretrained(
        source_path=args.codec_source_path,
        device=device,
        codec_source=args.codec_source,
        codec_source_revision=args.codec_source_revision,
        native_scaling_factor=args.native_scaling_factor,
    )
    # Store source path for save()
    codec._source_path = os.path.abspath(args.codec_source_path)
    print(f"    ✓ Codec loaded ({codec._codec_source})")

    # ── Step 2: Structural shape/range check ───────────────────────────────
    print("[2/6] Structural shape and range validation …")
    from codec.base import REQUIRED_LATENT_CHANNELS, REQUIRED_PIXEL_SIZE, REQUIRED_SPATIAL_FACTOR
    _structural_check(codec, device, REQUIRED_PIXEL_SIZE, REQUIRED_LATENT_CHANNELS, REQUIRED_SPATIAL_FACTOR)
    latent_size = REQUIRED_PIXEL_SIZE // codec.spatial_factor
    print(f"    ✓ Encode shape: (B,3,64,64) → (B,4,{latent_size},{latent_size})")
    print("    ✓ Decode range: [-1, 1]")
    print("    ✓ Round-trip latent shape preserved")

    # ── Step 3: Load CelebA splits ─────────────────────────────────────────
    print("[3/6] Loading CelebA dataset …")
    from config.config import DatasetConfig
    from data.celeba import get_dataloaders
    ds_cfg = DatasetConfig(
        name="celeba", root=args.celeba_root,
        image_size=REQUIRED_PIXEL_SIZE, num_workers=args.num_workers
    )
    train_loader, val_loader = get_dataloaders(ds_cfg, args.batch_size, seed=0)
    print(f"    ✓ Train batches: {len(train_loader)}")
    print(f"    ✓ Val   batches: {len(val_loader)}")

    # ── Step 4: Compute and freeze training-set statistics ─────────────────
    print("[4/6] Computing per-channel statistics on CelebA training set …")
    print("    (This encodes all training images — may take several minutes)")
    t0 = time.time()
    codec.compute_and_freeze_stats(train_loader, device)
    elapsed = time.time() - t0
    mean_vals = codec.latent_mean.flatten().tolist()
    std_vals = codec.latent_std.flatten().tolist()
    min_std = min(std_vals)
    print(f"    ✓ Done in {elapsed:.1f}s")
    print(f"    Per-channel mean : {[f'{v:.4f}' for v in mean_vals]}")
    print(f"    Per-channel std  : {[f'{v:.4f}' for v in std_vals]}")
    print(f"    Min channel std  : {min_std:.4f}")

    # Gate: minimum std
    STD_GATE = 0.1
    if min_std <= STD_GATE:
        print(f"\n  [FAIL] Min channel std {min_std:.4f} ≤ {STD_GATE} (project gate).")
        print("         This indicates a degenerate codec latent space.")
        return 1
    print(f"    ✓ Min std gate ({STD_GATE}) passed")

    # ── Step 5: Reconstruction quality on validation set ───────────────────
    print("[5/6] Evaluating reconstruction quality on 5,000 validation images …")
    fid_score, psnr_db, grid_images, recon_images = _eval_reconstruction(
        codec, val_loader, device,
        n_images=args.n_val_images,
        batch_size=args.batch_size,
    )
    print(f"    rFID : {fid_score:.3f}")
    print(f"    PSNR : {psnr_db:.2f} dB")

    # Project gates
    FID_GATE = 5.0
    PSNR_GATE = 30.0
    failures = []
    if fid_score >= FID_GATE:
        failures.append(f"rFID {fid_score:.3f} ≥ {FID_GATE} (project gate)")
    if psnr_db <= PSNR_GATE:
        failures.append(f"PSNR {psnr_db:.2f} dB ≤ {PSNR_GATE} dB (project gate)")
    # min_std already checked above

    # ── Step 6: Write outputs ───────────────────────────────────────────────
    print("[6/6] Writing outputs …")

    # Reconstruction grid
    grid_path = os.path.join(args.output_dir, "reconstruction_grid.png")
    if grid_images is not None:
        # Interleave originals and reconstructions: [orig1, recon1, orig2, recon2, …]
        n_grid = min(len(grid_images), len(recon_images), 16)
        interleaved = []
        for i in range(n_grid):
            interleaved.append(grid_images[i])
            interleaved.append(recon_images[i])
        grid_t = torch.stack(interleaved)
        save_image(grid_t, grid_path, normalize=True, value_range=(-1, 1), nrow=8)
        print(f"    ✓ Reconstruction grid → {grid_path}")

    # JSON report
    report = {
        "codec_source": codec._codec_source,
        "codec_source_revision": codec._codec_source_revision,
        "codec_weights_sha256": codec._codec_weights_sha256,
        "latent_channels": codec.latent_channels,
        "spatial_factor": codec.spatial_factor,
        "pixel_size": codec.pixel_size,
        "native_scaling_factor": codec._native_scaling_factor,
        "stats_frozen": True,
        "latent_mean_per_channel": mean_vals,
        "latent_std_per_channel": std_vals,
        "min_latent_std": min_std,
        "reconstruction_fid": fid_score,
        "reconstruction_psnr_db": psnr_db,
        "n_val_images": args.n_val_images,
        "gate_fid_threshold": FID_GATE,
        "gate_psnr_threshold_db": PSNR_GATE,
        "gate_min_std_threshold": STD_GATE,
        "gate_note": (
            "These thresholds are PROJECT-SPECIFIC GATES for this latent diffusion "
            "study and are NOT universal literature thresholds."
        ),
        "passed": len(failures) == 0,
        "failures": failures,
    }
    report_path = os.path.join(args.output_dir, "validation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"    ✓ Validation report → {report_path}")

    if failures:
        print(f"\n  [FAIL] Codec did not pass project gates:")
        for msg in failures:
            print(f"    • {msg}")
        print("\n  PROJECT GATE NOTE: These thresholds are specific to this study.")
        print("  Consult ANTIGRAVITY_PRETRAINED_LATENT_PLAN.md for the fallback plan.")
        return 1

    # Write the accepted codec checkpoint
    codec_path = os.path.join(args.output_dir, "accepted_codec.pt")
    codec.save(codec_path)
    print(f"    ✓ Codec checkpoint → {codec_path}")

    print(f"\n{'='*70}")
    print("  ALL GATES PASSED — codec accepted")
    print(f"  Accepted checkpoint : {codec_path}")
    print(f"{'='*70}\n")
    return 0


def _structural_check(codec, device, pixel_size, latent_channels, spatial_factor):
    """Shape and range assertions on a tiny synthetic batch."""
    import torch
    from codec.base import CodecCheckpointError

    with torch.no_grad():
        dummy_pixels = torch.zeros(2, 3, pixel_size, pixel_size, device=device)
        latents = codec.encode_mean(dummy_pixels)
        expected_lat_h = pixel_size // spatial_factor
        if latents.shape != (2, latent_channels, expected_lat_h, expected_lat_h):
            raise CodecCheckpointError(
                f"encode_mean() returned shape {tuple(latents.shape)}, "
                f"expected (2, {latent_channels}, {expected_lat_h}, {expected_lat_h})"
            )
        decoded = codec.decode(latents)
        if decoded.shape != (2, 3, pixel_size, pixel_size):
            raise CodecCheckpointError(
                f"decode() returned shape {tuple(decoded.shape)}, "
                f"expected (2, 3, {pixel_size}, {pixel_size})"
            )
        # Pixel range should be in [-1, 1]
        if decoded.min() < -1.01 or decoded.max() > 1.01:
            raise CodecCheckpointError(
                f"decode() output range [{decoded.min().item():.3f}, "
                f"{decoded.max().item():.3f}] is outside [-1, 1]. "
                f"Codec is not producing valid pixel values."
            )


def _eval_reconstruction(codec, val_loader, device, n_images, batch_size):
    """
    Compute rFID and PSNR for reconstructions on the validation set.
    Decodes in bounded batches — never holds all 5,000 decoder activations at once.
    """
    import torch
    from torchmetrics.image.fid import FrechetInceptionDistance

    fid_metric = FrechetInceptionDistance(normalize=False).to(device)
    psnr_sum = 0.0
    n_processed = 0
    grid_originals = []
    grid_recons = []

    with torch.no_grad():
        for images, _ in val_loader:
            if n_processed >= n_images:
                break
            images = images.to(device)
            # Crop to exactly n_images
            remaining = n_images - n_processed
            images = images[:remaining]

            # Encode → decode
            z = codec.encode_mean(images)
            recon = codec.decode(z)  # (B, 3, H, W) in [-1, 1]

            # FID: need uint8 [0,255]
            def _to_uint8(t):
                return ((t.clamp(-1, 1) + 1) / 2 * 255).to(torch.uint8)

            fid_metric.update(_to_uint8(images), real=True)
            fid_metric.update(_to_uint8(recon), real=False)

            # PSNR (in [-1, 1] space, so max_val = 2.0)
            mse = ((images - recon) ** 2).mean(dim=(1, 2, 3))
            psnr = 10 * torch.log10(4.0 / (mse + 1e-8))
            psnr_sum += psnr.sum().item()
            n_processed += images.shape[0]

            # Keep first 16 for the grid
            if len(grid_originals) < 16:
                take = min(16 - len(grid_originals), images.shape[0])
                grid_originals.extend(images[:take].cpu())
                grid_recons.extend(recon[:take].cpu())

    fid_score = fid_metric.compute().item()
    psnr_db = psnr_sum / max(n_processed, 1)
    return fid_score, psnr_db, grid_originals, grid_recons


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate a pretrained factor-8 AutoencoderKL and create a codec checkpoint."
    )
    p.add_argument("--codec-source", default="stabilityai/sd-vae-ft-mse",
                   help="Human-readable codec origin (recorded in checkpoint metadata)")
    p.add_argument("--codec-source-path", required=True,
                   help="Local path to the pretrained VAE directory (contains config.json)")
    p.add_argument("--codec-source-revision", default="local",
                   help="Revision to record; 'auto' reads source_manifest.json")
    p.add_argument("--native-scaling-factor", type=float, default=0.18215,
                   help="Native scaling factor applied to encoder outputs")
    p.add_argument("--celeba-root", default="./data/raw",
                   help="Root directory containing CelebA (auto-downloaded if absent)")
    p.add_argument("--output-dir", default="./results/codecs",
                   help="Directory for outputs (checkpoint, report, grid)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--n-val-images", type=int, default=5000,
                   help="Number of validation images for rFID/PSNR evaluation")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda", help="'cuda' or 'cpu'")
    return p.parse_args()


def _resolve_source_revision(source_path: str, requested: str) -> str:
    if requested != "auto":
        return requested
    manifest_path = os.path.join(source_path, "source_manifest.json")
    try:
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        revision = manifest["resolved_revision"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise SystemExit(
            f"--codec-source-revision auto requires a valid {manifest_path}: {exc}"
        ) from exc
    if not isinstance(revision, str) or not revision:
        raise SystemExit(f"Invalid resolved_revision in {manifest_path}")
    return revision


if __name__ == "__main__":
    sys.exit(_main())
