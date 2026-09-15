"""
Generic sampling engine. Contains ZERO algorithm-specific sampling
mathematics — it only calls algorithm.sample(n, nfe, device) and
measures wall-clock time / throughput / memory around that call.
"""
import os
from typing import List

import torch
from torchvision.utils import save_image

from algorithms.base import BaseAlgorithm
from utils.results import ResultRecord, ResultsWriter
from utils.timing import timer, peak_gpu_memory_mb, reset_peak_gpu_memory


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

    def run(self, n_samples: int, nfe_values: List[int], save_grid: bool = True,
            grid_size: int = 64) -> None:
        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")

        for module in self.algorithm.trainable_modules():
            module.eval()

        for nfe in nfe_values:
            if nfe < 1:
                raise ValueError(f"nfe must be >= 1, got {nfe}")

            torch.manual_seed(self.seed)  # identical noise/init across algorithms per NFE
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
                save_image(images[:grid_size], grid_path, normalize=True, value_range=(-1, 1))

    @torch.no_grad()
    def generate_for_evaluation(self, n_samples: int, nfe: int) -> torch.Tensor:
        """Used by evaluation/evaluator.py; not itself logged as a metric row."""
        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")
        if nfe < 1:
            raise ValueError(f"nfe must be >= 1, got {nfe}")

        for module in self.algorithm.trainable_modules():
            module.eval()
        torch.manual_seed(self.seed)
        return self.algorithm.sample(n_samples, nfe, self.device)
