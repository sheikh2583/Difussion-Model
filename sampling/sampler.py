"""
Generic sampling engine. Contains ZERO algorithm-specific sampling
mathematics — it only calls algorithm.sample(n, nfe, device) and
measures wall-clock time / throughput / memory around that call.

Latent extension (2026-09-21)
─────────────────────────────
Normalized latent tensors are NEVER passed to save_image(), including the
three-channel VQ-f4 representation. Representation is identified by the
backbone's sample_clamp policy, not channel count. Decoded reconstruction and sample
grids for latent experiments are produced separately by validate_codec.py
and scripts/smoke_latent.py.

The generate_for_evaluation() method is unchanged: it returns whatever
algorithm.sample() produces (normalised latents for latent configs) and
leaves decoding to the Evaluator.
"""
import logging
import os
from typing import List

import torch
from torchvision.utils import save_image

from algorithms.base import BaseAlgorithm
from utils.results import ResultRecord, ResultsWriter
from utils.timing import timer, peak_gpu_memory_mb, reset_peak_gpu_memory

_logger = logging.getLogger(__name__)


class Sampler:
    def __init__(self, algorithm: BaseAlgorithm, device: torch.device,
                 run_dir: str, experiment_name: str, seed: int):
        self.algorithm = algorithm
        self.device = device
        self.run_dir = run_dir
        self.seed = seed
        self.results = ResultsWriter(run_dir, experiment_name)
        self.samples_dir = os.path.join(run_dir, "samples")
        os.makedirs(self.samples_dir, exist_ok=True)

    def _fork_rng_devices(self):
        if self.device.type != "cuda":
            return []
        return [
            self.device.index
            if self.device.index is not None else torch.cuda.current_device()
        ]

    def run(self, n_samples: int, nfe_values: List[int], save_grid: bool = True,
            grid_size: int = 64) -> None:
        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")

        for module in self.algorithm.trainable_modules():
            module.eval()

        for nfe in nfe_values:
            if nfe < 1:
                raise ValueError(f"nfe must be >= 1, got {nfe}")

            # Evaluation gets deterministic noise without mutating the training
            # RNG stream used by subsequent epochs.
            with torch.random.fork_rng(devices=self._fork_rng_devices()):
                torch.manual_seed(self.seed)
                reset_peak_gpu_memory(self.device)

                with timer(self.device) as t:
                    images = self.algorithm.sample(n_samples, nfe, self.device)

            elapsed = t["elapsed"]
            peak_mem = peak_gpu_memory_mb(self.device)

            self.results.write(ResultRecord(
                algorithm=self.algorithm.name(),
                seed=self.seed,
                record_type="sampling",
                nfe=nfe,
                sampling_time=elapsed,
                time_per_image=elapsed / n_samples,
                images_per_second=n_samples / max(elapsed, 1e-8),  # elapsed==0 is a
                # measurement-precision floor, not a silently-changed experimental
                # setting, so this guard remains.
                peak_gpu_memory=peak_mem,
            ))

            if save_grid:
                grid_path = os.path.join(
                    self.samples_dir, f"{self.algorithm.name()}_nfe{nfe}.png")
                # ── Representation guard (latent experiment safety) ──────────
                # VQ-f4 latents also have three channels, so channel count alone
                # cannot distinguish them from RGB pixels.
                # Decoded RGB grids for latent experiments are written by
                # validate_codec.py (reconstruction grid) and
                # scripts/smoke_latent.py (sample grid with real codec).
                model_cfg = getattr(self.algorithm.model, "cfg", None)
                is_latent = not getattr(model_cfg, "sample_clamp", True)
                if is_latent or images.shape[1] != 3:
                    _logger.warning(
                        "Skipping grid save for NFE=%d: generated tensor has %d "
                        "channels and latent=%s. Normalized latent tensors must not "
                        "be saved directly as image grids. Decoded sample "
                        "grids are produced by scripts/smoke_latent.py.",
                        nfe,
                        images.shape[1],
                        is_latent,
                    )
                else:
                    save_image(
                        images[:grid_size], grid_path,
                        normalize=True, value_range=(-1, 1)
                    )

    @torch.no_grad()
    def generate_for_evaluation(
        self, n_samples: int, nfe: int, batch_size: int = 32
    ) -> torch.Tensor:
        """Generate in bounded accelerator batches and return CPU tensors.

        For pixel experiments: returns (N, 3, H, W) float32 in [-1, 1].
        For latent experiments: returns (N, C, H', W') float32 normalized latents.
        The Evaluator is responsible for decoding latents before FID/IS.
        """
        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")
        if nfe < 1:
            raise ValueError(f"nfe must be >= 1, got {nfe}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        for module in self.algorithm.trainable_modules():
            module.eval()
        batches = []
        with torch.random.fork_rng(devices=self._fork_rng_devices()):
            torch.manual_seed(self.seed)
            for start in range(0, n_samples, batch_size):
                count = min(batch_size, n_samples - start)
                batches.append(self.algorithm.sample(count, nfe, self.device).cpu())
        return torch.cat(batches, dim=0)
