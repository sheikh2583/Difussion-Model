#!/usr/bin/env python3
"""Benchmark every training flow with disposable synthetic batches.

This performs optimizer steps on newly initialized in-memory models only. It
does not consume training images, write checkpoints, or alter experiment runs.
The report is useful for checking memory fit and estimating wall-clock time on
the machine that will perform the real training.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import torch

from algorithms import ALGORITHM_REGISTRY
from config.config import ExperimentConfig
from data.dataset_registry import get_dataloaders_for_config
from models.backbone import build_backbone, count_parameters
from training.trainer import build_optimizer
from utils.gpu_lock import DEFAULT_LOCK_PATH, acquire_gpu_lock


CONFIGS = {
    "cifar10": {
        "fm": "config/fm_full.json",
        "fm_lognorm": "config/fm_lognorm_full.json",
        "mf": "config/mf_full.json",
        "consistency": "config/consistency_full.json",
        "mf_distill": "config/mf_distill_full.json",
        "reflow": "config/reflow_full.json",
    },
    "celeba": {
        "fm": "config/fm_celeba64.json",
        "fm_lognorm": "config/fm_lognorm_celeba64.json",
        "mf": "config/mf_celeba64.json",
        "consistency": "config/consistency_celeba64.json",
        "mf_distill": "config/mf_distill_celeba64.json",
        "reflow": "config/reflow_celeba64.json",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["all", *CONFIGS], default="all")
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--measure-steps", type=int, default=3)
    parser.add_argument("--sample-batch-size", type=int, default=16)
    parser.add_argument("--lock-file", default=str(DEFAULT_LOCK_PATH))
    parser.add_argument(
        "--output", default="results/training_flow_benchmark.json"
    )
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def disposable_dependencies(cfg: ExperimentConfig, temp_root: Path) -> dict:
    kwargs = copy.deepcopy(cfg.algorithm_kwargs)
    if "teacher_checkpoint" in kwargs and not Path(kwargs["teacher_checkpoint"]).is_file():
        teacher_path = temp_root / f"teacher_{cfg.dataset.image_size}.pt"
        if not teacher_path.is_file():
            teacher = build_backbone(
                cfg.backbone, image_size=cfg.dataset.image_size
            )
            torch.save({"model_state": teacher.state_dict()}, teacher_path)
            del teacher
        kwargs["teacher_checkpoint"] = str(teacher_path)
    if "pairs_path" in kwargs:
        pairs_path = temp_root / f"pairs_{cfg.dataset.image_size}.pt"
        if not pairs_path.is_file():
            shape = (
                max(2, cfg.batch_size),
                cfg.backbone.in_channels,
                cfg.dataset.image_size,
                cfg.dataset.image_size,
            )
            torch.save(
                {"z1": torch.randn(shape), "x0": torch.randn(shape).clamp(-1, 1)},
                pairs_path,
            )
        kwargs["pairs_path"] = str(pairs_path)
    return kwargs


def benchmark_one(
    algorithm_key: str,
    config_path: str,
    device: torch.device,
    dataset_size: int,
    temp_root: Path,
    warmup_steps: int,
    measure_steps: int,
    sample_batch_size: int,
) -> dict:
    cfg = ExperimentConfig.load(config_path)
    kwargs = disposable_dependencies(cfg, temp_root)
    model = build_backbone(cfg.backbone, image_size=cfg.dataset.image_size)
    algorithm = ALGORITHM_REGISTRY[algorithm_key](model, algorithm_kwargs=kwargs)
    modules = algorithm.trainable_modules()
    for module in modules:
        module.to(device)
        module.train()
    optimizer = build_optimizer(modules, cfg)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=cfg.amp and device.type == "cuda"
    )
    batch = torch.randn(
        cfg.batch_size,
        cfg.backbone.in_channels,
        cfg.dataset.image_size,
        cfg.dataset.image_size,
        device=device,
    ).clamp(-1, 1)

    def step() -> float:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            enabled=cfg.amp and device.type == "cuda",
        ):
            loss = algorithm.training_step(batch)["loss"]
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss: {loss.item()}")
        scaler.scale(loss).backward()
        scale_before_step = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= scale_before_step:
            algorithm.on_after_optimizer_step()
        return float(loss.detach())

    for _ in range(warmup_steps):
        step()
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    losses = [step() for _ in range(measure_steps)]
    synchronize(device)
    step_seconds = (time.perf_counter() - started) / measure_steps
    peak_mb = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        if device.type == "cuda" else 0.0
    )

    optimizer.zero_grad(set_to_none=True)
    del batch, optimizer, scaler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    for module in modules:
        module.eval()
    with torch.no_grad():
        warm_samples = algorithm.sample(sample_batch_size, 1, device)
    del warm_samples
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.no_grad():
        samples = algorithm.sample(sample_batch_size, 1, device)
    synchronize(device)
    sample_seconds = time.perf_counter() - started
    sample_peak_mb = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        if device.type == "cuda" else 0.0
    )
    assert samples.shape[0] == sample_batch_size and torch.isfinite(samples).all()

    steps_per_epoch = dataset_size // cfg.batch_size
    estimated_training_hours = step_seconds * steps_per_epoch * cfg.epochs / 3600
    total_params, trainable_params = count_parameters(model)
    result = {
        "algorithm": algorithm_key,
        "dataset": cfg.dataset.name,
        "config": config_path,
        "batch_size": cfg.batch_size,
        "epochs": cfg.epochs,
        "dataset_size": dataset_size,
        "steps_per_epoch": steps_per_epoch,
        "step_seconds": step_seconds,
        "peak_training_memory_mb": peak_mb,
        "losses": losses,
        "nfe1_seconds_per_image": sample_seconds / sample_batch_size,
        "peak_sampling_memory_mb": sample_peak_mb,
        "compute_only_training_hours": estimated_training_hours,
        "backbone_parameters": total_params,
        "trainable_backbone_parameters": trainable_params,
    }
    del samples, algorithm, modules, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> int:
    args = parse_args()
    if args.warmup_steps < 0 or args.measure_steps < 1:
        raise ValueError("warmup steps must be >= 0 and measure steps must be >= 1")
    if args.sample_batch_size < 1:
        raise ValueError("sample batch size must be >= 1")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for representative benchmarking.")
    device = torch.device("cuda")
    datasets = list(CONFIGS) if args.dataset == "all" else [args.dataset]
    dataset_sizes = {}
    for dataset_name in datasets:
        first_config = ExperimentConfig.load(next(iter(CONFIGS[dataset_name].values())))
        train_loader, _ = get_dataloaders_for_config(first_config)
        dataset_sizes[dataset_name] = len(train_loader.dataset)
        del train_loader

    report = {
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "warmup_steps": args.warmup_steps,
        "measure_steps": args.measure_steps,
        "results": [],
    }
    with tempfile.TemporaryDirectory(prefix="diffusion-flow-benchmark-") as temp:
        with acquire_gpu_lock(args.lock_file, command="benchmark_training_flows.py"):
            for dataset_name in datasets:
                for algorithm_key, config_path in CONFIGS[dataset_name].items():
                    print(f"Benchmarking {algorithm_key}/{dataset_name}...", flush=True)
                    try:
                        row = benchmark_one(
                            algorithm_key,
                            config_path,
                            device,
                            dataset_sizes[dataset_name],
                            Path(temp),
                            args.warmup_steps,
                            args.measure_steps,
                            args.sample_batch_size,
                        )
                        row["status"] = "ok"
                    except Exception as exc:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        row = {
                            "algorithm": algorithm_key,
                            "dataset": dataset_name,
                            "config": config_path,
                            "status": (
                                "oom" if isinstance(exc, torch.OutOfMemoryError)
                                else "error"
                            ),
                            "error": str(exc),
                        }
                    report["results"].append(row)
                    print(json.dumps(row, sort_keys=True), flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report: {output.resolve()}")
    return 1 if any(row["status"] != "ok" for row in report["results"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
