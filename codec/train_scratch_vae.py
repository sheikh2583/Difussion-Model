"""Operator-only training command for the scratch factor-4 KL-VAE.

The baseline objective is pixel MSE plus KL. Pixel MSE can produce blurry
reconstructions and may fail the strict project perceptual gate; changing to a
perceptual or adversarial objective is an experiment change requiring approval.

Example (run only after the pretrained codec is rejected):
    ./scripts/linux/train_scratch_codec.sh

The recommended baseline is AdamW at 1e-4, batch 128, 60 epochs, AMP,
gradient clipping at 1.0, and a linear KL weight warmup from 1e-5 to 1e-4
over the first 20 epochs. Every setting is available as a CLI option.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from codec.scratch_vae import ScratchKLVAE
from config.config import DatasetConfig
from data.celeba import get_datasets


# Recommended first run for the 64x64 factor-4 baseline. These are named
# constants so checkpoints, --help, tests, and the Linux launcher agree.
DEFAULT_BATCH_SIZE = 128
DEFAULT_EPOCHS = 60
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_KL_START = 1e-5
DEFAULT_KL_END = 1e-4
DEFAULT_KL_WARMUP_EPOCHS = 20
DEFAULT_GRAD_CLIP_NORM = 1.0
DEFAULT_VALIDATE_EVERY = 5
DEFAULT_VALIDATION_SAMPLES = 5000


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the scratch CelebA KL-f4 fallback")
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--work-dir", default="results/scratch_vae")
    parser.add_argument(
        "--output", default="results/codecs/accepted_scratch_kl_vae.pt"
    )
    parser.add_argument(
        "--resume", default=None,
        help="Resumable checkpoint path, or 'auto' for the newest checkpoint in work-dir",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--kl-start", type=float, default=DEFAULT_KL_START)
    parser.add_argument("--kl-end", type=float, default=DEFAULT_KL_END)
    parser.add_argument(
        "--kl-warmup-epochs", type=int, default=DEFAULT_KL_WARMUP_EPOCHS
    )
    parser.add_argument(
        "--gradient-clip-norm", type=float, default=DEFAULT_GRAD_CLIP_NORM,
        help="Set to 0 to disable gradient clipping",
    )
    parser.add_argument("--validate-every", type=int, default=DEFAULT_VALIDATE_EVERY)
    parser.add_argument(
        "--validation-samples", type=int, default=DEFAULT_VALIDATION_SAMPLES
    )
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True,
        help="Use CUDA automatic mixed precision (recommended on RTX GPUs)",
    )
    args = parser.parse_args(argv)
    positive_ints = {
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "kl_warmup_epochs": args.kl_warmup_epochs,
        "validate_every": args.validate_every,
        "validation_samples": args.validation_samples,
    }
    for name, value in positive_ints.items():
        if value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if args.weight_decay < 0 or args.kl_start < 0 or args.kl_end < 0:
        parser.error("weight decay and KL weights must be non-negative")
    if args.gradient_clip_norm < 0:
        parser.error("--gradient-clip-norm must be non-negative")
    return args


def _atomic_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _kl_weight(
    epoch_index: int,
    start: float = DEFAULT_KL_START,
    end: float = DEFAULT_KL_END,
    warmup_epochs: int = DEFAULT_KL_WARMUP_EPOCHS,
) -> float:
    progress = min(max(epoch_index, 0), warmup_epochs) / float(warmup_epochs)
    return start + progress * (end - start)


def _latest_checkpoint(work_dir: Path) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    pattern = re.compile(r"^scratch_vae_epoch([1-9][0-9]*)\.pt$")
    for path in (work_dir / "checkpoints").glob("scratch_vae_epoch*.pt"):
        match = pattern.match(path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def _hyperparameters(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "kl_start": args.kl_start,
        "kl_end": args.kl_end,
        "kl_warmup_epochs": args.kl_warmup_epochs,
        "gradient_clip_norm": args.gradient_clip_norm,
        "amp": args.amp,
        "seed": args.seed,
    }


def _validate_resume_hyperparameters(
    state: dict[str, Any], requested: dict[str, Any], path: Path
) -> None:
    """Prevent a resume from silently changing the optimization experiment."""
    stored = state.get("hyperparameters")
    if isinstance(stored, dict):
        immutable = (
            "batch_size", "learning_rate", "weight_decay", "kl_start", "kl_end",
            "kl_warmup_epochs", "gradient_clip_norm", "seed",
        )
        mismatches = [
            f"{key}: checkpoint={stored[key]!r}, requested={requested[key]!r}"
            for key in immutable
            if key in stored and stored[key] != requested[key]
        ]
        if mismatches:
            raise ValueError(
                f"Resume hyperparameters do not match {path}: " + "; ".join(mismatches)
            )
        return

    # Compatibility for checkpoints produced by the original trainer, which
    # did not embed its fixed hyperparameters.
    groups = state.get("optimizer_state", {}).get("param_groups", [])
    if groups:
        old_lr = float(groups[0].get("lr", DEFAULT_LEARNING_RATE))
        old_wd = float(groups[0].get("weight_decay", DEFAULT_WEIGHT_DECAY))
        if old_lr != requested["learning_rate"] or old_wd != requested["weight_decay"]:
            raise ValueError(
                f"Legacy resume checkpoint {path} used lr={old_lr:g}, "
                f"weight_decay={old_wd:g}; requested lr={requested['learning_rate']:g}, "
                f"weight_decay={requested['weight_decay']:g}"
            )


@torch.no_grad()
def validate(
    model: ScratchKLVAE,
    loader: DataLoader,
    device: torch.device,
    limit: int = 5000,
) -> dict[str, Any]:
    # Keep the heavyweight torchmetrics/Inception import out of CLI parsing and
    # training startup. It is needed only at the five-epoch validation points.
    from evaluation.metrics import compute_fid

    model.eval()
    real_batches: list[torch.Tensor] = []
    reconstructed_batches: list[torch.Tensor] = []
    squared_error = 0.0
    pixel_count = 0
    latent_sum = torch.zeros(4, dtype=torch.float64, device=device)
    latent_sq_sum = torch.zeros_like(latent_sum)
    latent_count = 0
    collected = 0

    for images, _ in loader:
        if collected >= limit:
            break
        images = images[: limit - collected].to(device, non_blocking=True)
        means = model.encode_mean(images)
        reconstructions = model.decode(means)
        real_batches.append(images.cpu())
        reconstructed_batches.append(reconstructions.cpu())
        difference = (reconstructions - images).double()
        squared_error += difference.square().sum().item()
        pixel_count += images.numel()
        latent_sum += means.double().sum(dim=(0, 2, 3))
        latent_sq_sum += means.double().square().sum(dim=(0, 2, 3))
        latent_count += means.shape[0] * means.shape[2] * means.shape[3]
        collected += images.shape[0]

    if collected != limit:
        raise RuntimeError(f"Validation requires {limit} images, but loader yielded {collected}")
    mse = squared_error / pixel_count
    psnr = float("inf") if mse == 0 else 10.0 * math.log10(4.0 / mse)
    latent_mean = latent_sum / latent_count
    latent_std = (latent_sq_sum / latent_count - latent_mean.square()).clamp_min(0).sqrt()
    rfid = compute_fid(
        torch.cat(real_batches), torch.cat(reconstructed_batches), device
    )
    return {
        "rfid": rfid,
        "psnr": psnr,
        "latent_mean": latent_mean.cpu().tolist(),
        "latent_std": latent_std.cpu().tolist(),
    }


def _print_gate(metrics: dict[str, Any]) -> bool:
    minimum_std = min(metrics["latent_std"])
    passed = metrics["rfid"] < 5.0 and metrics["psnr"] > 30.0 and minimum_std > 0.1
    print("\n=== PROJECT ACCEPTANCE GATE (strict project threshold) ===")
    print(f"rFID < 5:             {metrics['rfid']:.4f}")
    print(f"PSNR > 30 dB:         {metrics['psnr']:.4f} dB")
    print(f"minimum latent std > .1: {minimum_std:.6f}")
    print(f"GATE RESULT: {'PASS' if passed else 'FAIL'}")
    print("This MSE+KL baseline may be blurry; no perceptual/adversarial loss was added.")
    return passed


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")
    _seed_everything(args.seed)

    dataset_cfg = DatasetConfig(root=args.data_root, image_size=64, num_workers=args.num_workers)
    train_set, valid_set = get_datasets(dataset_cfg)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda", generator=generator,
    )
    valid_loader = DataLoader(
        valid_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )

    model = ScratchKLVAE().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    hyperparameters = _hyperparameters(args)
    start_epoch = 0
    work_dir = Path(args.work_dir)
    resume_path: Path | None = None
    if args.resume == "auto":
        resume_path = _latest_checkpoint(work_dir)
        if resume_path is None:
            raise FileNotFoundError(
                f"--resume auto found no checkpoint under {work_dir / 'checkpoints'}"
            )
    elif args.resume:
        resume_path = Path(args.resume)
    if resume_path is not None:
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        _validate_resume_hyperparameters(state, hyperparameters, resume_path)
        model.load_state_dict(state["model_state"], strict=True)
        optimizer.load_state_dict(state["optimizer_state"])
        if "scaler_state" in state:
            scaler.load_state_dict(state["scaler_state"])
        start_epoch = int(state["epoch"])
        if start_epoch >= args.epochs:
            raise ValueError(
                f"Resume checkpoint is at epoch {start_epoch}, but --epochs={args.epochs}"
            )
        if "generator_state" in state:
            generator.set_state(state["generator_state"])
        rng_state = state.get("rng_state")
        if rng_state:
            random.setstate(rng_state["python"])
            np.random.set_state(rng_state["numpy"])
            torch.set_rng_state(rng_state["torch"])
            if torch.cuda.is_available() and rng_state.get("cuda"):
                torch.cuda.set_rng_state_all(rng_state["cuda"])
        print(f"Resuming from {resume_path} at completed epoch {start_epoch}", flush=True)

    checkpoint_dir = work_dir / "checkpoints"
    _atomic_write_json(hyperparameters, work_dir / "hyperparameters.json")
    print(
        "Scratch KL-VAE settings: "
        f"batch={args.batch_size}, epochs={args.epochs}, lr={args.learning_rate:g}, "
        f"weight_decay={args.weight_decay:g}, KL={args.kl_start:g}->{args.kl_end:g} "
        f"over {args.kl_warmup_epochs} epochs, grad_clip={args.gradient_clip_norm:g}, "
        f"amp={use_amp}",
        flush=True,
    )
    final_metrics: dict[str, Any] | None = None
    for epoch_index in range(start_epoch, args.epochs):
        model.train()
        beta = _kl_weight(
            epoch_index, args.kl_start, args.kl_end, args.kl_warmup_epochs
        )
        running = 0.0
        for images, _ in train_loader:
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                sample, mean, logvar = model.encode(images)
                reconstruction = model.decode(sample)
                reconstruction_loss = F.mse_loss(reconstruction, images)
                kl = model.kl_loss(mean, logvar)
                loss = reconstruction_loss + beta * kl
            scaler.scale(loss).backward()
            if args.gradient_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()

        completed_epoch = epoch_index + 1
        print(
            f"epoch={completed_epoch}/{args.epochs} beta={beta:.8f} "
            f"loss={running / max(len(train_loader), 1):.6f}", flush=True
        )
        if completed_epoch % args.validate_every == 0 or completed_epoch == args.epochs:
            checkpoint = {
                "epoch": completed_epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "generator_state": generator.get_state(),
                "seed": args.seed,
                "hyperparameters": hyperparameters,
                "rng_state": {
                    "python": random.getstate(),
                    "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                },
            }
            _atomic_save(checkpoint, checkpoint_dir / f"scratch_vae_epoch{completed_epoch}.pt")
            final_metrics = validate(
                model, valid_loader, device, limit=args.validation_samples
            )
            _atomic_write_json(
                {"epoch": completed_epoch, **final_metrics},
                work_dir / "metrics" / f"epoch_{completed_epoch:03d}.json",
            )
            _print_gate(final_metrics)

    if final_metrics is None:
        final_metrics = validate(
            model, valid_loader, device, limit=args.validation_samples
        )
    if not _print_gate(final_metrics):
        raise SystemExit(2)

    # The generative cache must use statistics frozen over posterior means from
    # the complete training split, never validation or posterior samples.
    stats_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    model.compute_and_freeze_stats(stats_loader, device)
    model.save(args.output)
    print(f"Accepted factory-loadable scratch codec saved to {args.output}")


if __name__ == "__main__":
    main()
