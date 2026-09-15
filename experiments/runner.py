"""
Generic experiment runner: train -> checkpoint -> sample -> evaluate,
for whichever algorithm is passed in. Contains no FM/MF-specific
assumptions, and actively enforces the experimental-fairness
constraints agreed on:

  - algorithm_kwargs must not shadow any protected shared field;
  - backbone/model is built once, from the shared config, for any
    algorithm;
  - dataloaders are built once from the shared config;
  - FID reference statistics are cached and reused across algorithms.
"""
import os
from typing import Type

import torch

from algorithms.base import BaseAlgorithm
from config.config import ExperimentConfig
from data.cifar10 import get_dataloaders
from evaluation.evaluator import Evaluator, ensure_fid_reference
from models.backbone import build_backbone
from sampling.sampler import Sampler
from training.trainer import Trainer
from utils.device import resolve_device


def assert_no_protected_key_override(cfg: ExperimentConfig) -> None:
    """
    Enforces constraint: algorithm_kwargs may only contain parameters
    inherently required by the algorithm; it must never contain a key
    that shadows a shared experimental control.
    """
    overlap = set(cfg.algorithm_kwargs.keys()) & set(cfg._protected_keys)
    if overlap:
        raise ValueError(
            f"algorithm_kwargs illegally overrides shared experimental "
            f"controls: {sorted(overlap)}. Shared controls must be set "
            f"only via the top-level ExperimentConfig fields."
        )


class ExperimentRunner:
    def __init__(self, cfg: ExperimentConfig, algorithm_cls: Type[BaseAlgorithm]):
        assert_no_protected_key_override(cfg)

        self.cfg = cfg
        self.algorithm_cls = algorithm_cls
        self.device = resolve_device(cfg)

        self.run_dir = os.path.join(cfg.output_dir, cfg.experiment_name)
        os.makedirs(self.run_dir, exist_ok=True)

        self.train_loader, self.test_loader = get_dataloaders(
            cfg.dataset, cfg.batch_size, cfg.seed)

        model = build_backbone(cfg.backbone, image_size=cfg.dataset.image_size)
        self.algorithm = algorithm_cls(model, algorithm_kwargs=cfg.algorithm_kwargs)

        self.evaluator = Evaluator(cfg, self.run_dir, self.device)
        self.sampler = Sampler(self.algorithm, self.device, self.run_dir,
                                cfg.experiment_name, cfg.seed)

    def _eval_hook(self, epoch: int) -> None:
        self.evaluator.evaluate(self.sampler, nfe_values=self.cfg.evaluation.nfe_values)

    def train(self) -> Trainer:
        ensure_fid_reference(self.cfg, self.test_loader, self.device)

        trainer = Trainer(
            algorithm=self.algorithm,
            train_loader=self.train_loader,
            cfg=self.cfg,
            run_dir=self.run_dir,
            device=self.device,
            eval_hook=self._eval_hook,
        )
        trainer.fit()
        return trainer

    def sample(self) -> None:
        self.sampler.run(
            n_samples=self.cfg.evaluation.num_generated_samples,
            nfe_values=self.cfg.evaluation.nfe_values,
        )

    def evaluate(self) -> None:
        ensure_fid_reference(self.cfg, self.test_loader, self.device)
        self.evaluator.evaluate(self.sampler, nfe_values=self.cfg.evaluation.nfe_values)

    def run_full(self) -> None:
        self.train()
        self.sample()
        self.evaluate()
