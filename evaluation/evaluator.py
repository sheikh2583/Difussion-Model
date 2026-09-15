"""
Common evaluation pipeline. Does not need to know which algorithm
produced the images — it only consumes algorithm.sample() output via
the Sampler, plus the cached real-data reference statistics.
"""
import json
import os
from typing import List

import torch
from torch.utils.data import DataLoader

from config.config import ExperimentConfig
from evaluation.metrics import cache_real_images, load_real_images, compute_fid, compute_inception_score
from sampling.sampler import Sampler
from utils.results import ResultRecord, ResultsWriter


def _fid_reference_meta_path(cache_path: str) -> str:
    return cache_path + ".meta.json"


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


class Evaluator:
    def __init__(self, cfg: ExperimentConfig, run_dir: str, device: torch.device):
        self.cfg = cfg
        self.run_dir = run_dir
        self.device = device
        self.results = ResultsWriter(run_dir, cfg.experiment_name)

    def evaluate(self, sampler: Sampler, nfe_values: List[int]) -> None:
        real_images = None
        if "fid" in self.cfg.evaluation.metrics:
            real_images = load_real_images(
                self.cfg.evaluation.fid_reference_cache, self.device)

        for nfe in nfe_values:
            images = sampler.generate_for_evaluation(
                self.cfg.evaluation.num_generated_samples, nfe)

            fid_score = None
            if real_images is not None:
                fid_score = compute_fid(real_images, images, self.device)

            is_mean = is_std = None
            if "is" in self.cfg.evaluation.metrics:
                is_mean, is_std = compute_inception_score(images, self.device)

            self.results.write(ResultRecord(
                algorithm=sampler.algorithm.name(),
                seed=sampler.seed,
                record_type="evaluation",
                nfe=nfe,
                fid=fid_score,
                is_mean=is_mean,
                is_std=is_std,
            ))
