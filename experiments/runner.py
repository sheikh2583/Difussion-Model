"""
Generic experiment runner. Updated to use dataset_registry so any
registered dataset (cifar10, celeba, ...) works without further changes.
All other logic identical to original.
"""
import os
from typing import Type

import torch

from algorithms.base import BaseAlgorithm
from algorithms import ALGORITHM_REGISTRY
from config.config import ExperimentConfig
from data.dataset_registry import get_dataloaders_for_config
from evaluation.evaluator import Evaluator, ensure_fid_reference
from models.backbone import build_backbone
from sampling.sampler import Sampler
from training.trainer import Trainer
from utils.device import resolve_device


def assert_no_protected_key_override(cfg: ExperimentConfig) -> None:
    overlap = set(cfg.algorithm_kwargs.keys()) & set(cfg._protected_keys)
    if overlap:
        raise ValueError(
            f"algorithm_kwargs illegally overrides shared experimental "
            f"controls: {sorted(overlap)}."
        )


class ExperimentRunner:
    def __init__(self, cfg: ExperimentConfig, algorithm_cls: Type[BaseAlgorithm]):
        assert_no_protected_key_override(cfg)

        self.cfg = cfg
        self.algorithm_cls = algorithm_cls
        self.device = resolve_device(cfg)

        self.run_dir = os.path.join(cfg.output_dir, cfg.experiment_name)
        os.makedirs(self.run_dir, exist_ok=True)

        # Dataset registry replaces direct cifar10 import
        self.train_loader, self.test_loader = get_dataloaders_for_config(cfg)

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

    def run_full(self) -> None:
        self.train()
        self.evaluator.evaluate(self.sampler, nfe_values=self.cfg.evaluation.nfe_values)
