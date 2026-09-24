#!/usr/bin/env python3
"""Generate images with adaptive-NFE or multiscale Mean Flow sampling."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import torch
from torchvision.utils import save_image

from algorithms.mean_flow import MeanFlowAlgorithm
from algorithms.mean_flow_adaptive_nfe import AdaptiveMeanFlowSampler
from algorithms.mean_flow_multiscale import MultiScaleMeanFlowPipeline
from codec.codec_factory import load_codec
from config.config import ExperimentConfig
from models.backbone import build_backbone
from utils.checkpoints import load_algorithm_state
from utils.device import resolve_device


def add_checkpoint_arguments(parser: argparse.ArgumentParser, prefix: str = "") -> None:
    option_prefix = f"{prefix}-" if prefix else ""
    destination_prefix = f"{prefix}_" if prefix else ""
    parser.add_argument(
        f"--{option_prefix}config", required=True, dest=f"{destination_prefix}config"
    )
    parser.add_argument(
        f"--{option_prefix}checkpoint",
        required=True,
        dest=f"{destination_prefix}checkpoint",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    adaptive = subparsers.add_parser("adaptive")
    add_checkpoint_arguments(adaptive)
    adaptive.add_argument("--n-samples", type=int, default=64)
    adaptive.add_argument("--min-nfe", type=int, default=2)
    adaptive.add_argument("--max-nfe", type=int, default=4)
    adaptive.add_argument("--threshold", type=float, default=0.05)
    adaptive.add_argument("--codec", default=None)
    adaptive.add_argument("--output", default="results/adaptive_mean_flow.png")

    multiscale = subparsers.add_parser("multiscale")
    add_checkpoint_arguments(multiscale)
    multiscale.add_argument("--n-samples", type=int, default=64)
    multiscale.add_argument("--coarse-nfe", type=int, default=1)
    multiscale.add_argument("--fine-nfe", type=int, default=4)
    multiscale.add_argument("--t-renoise", type=float, default=0.3)
    multiscale.add_argument("--codec", default=None)
    multiscale.add_argument("--output", default="results/multiscale_mean_flow.png")
    return parser.parse_args()


def load_mean_flow(config_path: str, checkpoint_path: str, device: torch.device):
    cfg = ExperimentConfig.load(config_path)
    model = build_backbone(cfg.backbone, image_size=cfg.dataset.image_size)
    algorithm = MeanFlowAlgorithm(model, algorithm_kwargs=cfg.algorithm_kwargs)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    load_algorithm_state(algorithm, payload)
    for module in algorithm.trainable_modules():
        module.to(device)
        module.eval()
    return cfg, algorithm


def save_outputs(images: torch.Tensor, output: str, metadata: dict) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(
        images,
        str(output_path),
        nrow=max(1, int(len(images) ** 0.5)),
        normalize=True,
        value_range=(-1, 1),
    )
    metadata_path = output_path.with_suffix(output_path.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Images:   {output_path}")
    print(f"Metadata: {metadata_path}")


def load_latent_codec(cfg, override: str | None, device: torch.device):
    path = override or cfg.dataset.codec_checkpoint
    if not path:
        raise ValueError("A frozen latent codec checkpoint is required for image output")
    return load_codec(path, device, require_frozen=True)


def main() -> int:
    args = parse_args()
    if args.mode == "adaptive":
        cfg = ExperimentConfig.load(args.config)
        device = resolve_device(cfg)
        _, algorithm = load_mean_flow(args.config, args.checkpoint, device)
        codec = load_latent_codec(cfg, args.codec, device)
        sampler = AdaptiveMeanFlowSampler(
            algorithm,
            min_nfe=args.min_nfe,
            max_nfe=args.max_nfe,
            confidence_threshold=args.threshold,
            codec=codec,
        )
        images = sampler.sample(args.n_samples, device)
        save_outputs(images, args.output, {
            "mode": "adaptive",
            "n_samples": args.n_samples,
            "min_nfe": args.min_nfe,
            "max_nfe": args.max_nfe,
            "confidence_threshold": args.threshold,
            "average_nfe": sampler.last_average_nfe,
            "nfe_per_sample": sampler.last_nfe_per_sample.tolist(),
        })
        return 0

    cfg = ExperimentConfig.load(args.config)
    device = resolve_device(cfg)
    _, algorithm = load_mean_flow(args.config, args.checkpoint, device)
    codec = load_latent_codec(cfg, args.codec, device)
    pipeline = MultiScaleMeanFlowPipeline(
        algorithm,
        coarse_nfe=args.coarse_nfe,
        fine_nfe=args.fine_nfe,
        t_renoise=args.t_renoise,
        codec=codec,
    )
    images = pipeline.sample(args.n_samples, device)
    save_outputs(images, args.output, {
        "mode": "multiscale",
        "n_samples": args.n_samples,
        "coarse_nfe": args.coarse_nfe,
        "fine_nfe": args.fine_nfe,
        "total_nfe": args.coarse_nfe + args.fine_nfe,
        "t_renoise": args.t_renoise,
        "latent_size": cfg.dataset.image_size,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
