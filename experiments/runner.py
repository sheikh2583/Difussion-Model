"""
Generic experiment runner. Updated to use dataset_registry so any
registered dataset (cifar10, celeba, ...) works without further changes.
All other logic identical to original.
"""
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Type

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
from utils.checkpoint_provenance import build_provenance, validate_checkpoint_file
from utils.checkpoint_runs import checkpoint_run_directory, latest_epoch_checkpoint
from utils.seed import set_seed
from utils.run_environment import collect_run_environment, write_run_environment


def assert_no_protected_key_override(cfg: ExperimentConfig) -> None:
    overlap = set(cfg.algorithm_kwargs.keys()) & set(cfg._protected_keys)
    if overlap:
        raise ValueError(
            f"algorithm_kwargs illegally overrides shared experimental "
            f"controls: {sorted(overlap)}."
        )


class ExperimentRunner:
    def __init__(self, cfg: ExperimentConfig, algorithm_cls: Type[BaseAlgorithm],
                 algorithm_key: Optional[str] = None,
                 checkpoint_run_number: int = 1,
                 machine_label: Optional[str] = None):
        assert_no_protected_key_override(cfg)

        self.cfg = cfg
        self.algorithm_cls = algorithm_cls
        self.algorithm_key = algorithm_key or next(
            (key for key, value in ALGORITHM_REGISTRY.items() if value is algorithm_cls),
            None,
        )
        self.checkpoint_provenance = build_provenance(
            cfg, algorithm_cls, self.algorithm_key
        )
        self.checkpoint_run_number = checkpoint_run_number
        self.device = resolve_device(cfg)

        # Seed before constructing the backbone so independent algorithm runs
        # begin from the same reproducible initialization.
        set_seed(cfg.seed)

        # Derive run directory: <output_dir>/<experiment_name>_<dataset.name>
        # This keeps source code dataset-agnostic — changing the dataset in the
        # config automatically routes outputs to a different directory.
        run_name = f"{cfg.experiment_name}_{cfg.dataset.name}"
        self.run_dir = os.path.join(cfg.output_dir, run_name)
        os.makedirs(self.run_dir, exist_ok=True)
        self.run_environment = collect_run_environment(
            Path(__file__).resolve().parent.parent,
            machine_label=machine_label,
        )
        serialized_config = json.dumps(
            asdict(cfg), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        self.run_environment["config_sha256"] = hashlib.sha256(
            serialized_config
        ).hexdigest()
        write_run_environment(Path(self.run_dir), self.run_environment)

        # Dataset registry replaces direct cifar10 import
        self.train_loader, self.test_loader = get_dataloaders_for_config(cfg)

        model = build_backbone(cfg.backbone, image_size=cfg.dataset.image_size)
        self.algorithm = algorithm_cls(model, algorithm_kwargs=cfg.algorithm_kwargs)

        self.evaluator = Evaluator(cfg, self.run_dir, self.device)
        self.sampler = Sampler(self.algorithm, self.device, self.run_dir,
                                run_name, cfg.seed)

    def _eval_hook(self, epoch: int) -> None:
        checkpoint_dir = checkpoint_run_directory(Path(self.run_dir), self.checkpoint_run_number)
        checkpoint_path = checkpoint_dir / f"{self.algorithm.name()}_epoch{epoch}.pt"
        self.evaluator.evaluate(
            self.sampler,
            nfe_values=self.cfg.evaluation.nfe_values,
            checkpoint_path=str(checkpoint_path)
        )

    def _resolve_resume_checkpoint(self, requested: str) -> str:
        if requested != "auto":
            path = Path(requested)
            if not path.is_file():
                raise FileNotFoundError(f"Resume checkpoint not found: {path}")
            return str(path)

        latest = latest_epoch_checkpoint(
            Path(self.run_dir),
            class_name=self.algorithm.name(),
            maximum_epoch=self.cfg.epochs,
            run_number=self.checkpoint_run_number,
        )
        if latest is None:
            raise FileNotFoundError(
                f"Cannot continue {self.run_dir}: no checkpoint matching "
                f"checkpoints/run_{self.checkpoint_run_number}/"
                f"{self.algorithm.name()}_epoch<N>.pt was found. Use --mode fresh "
                "to preserve the directory and restart safely."
            )
        return str(latest[1])

    def train(self, resume_checkpoint: Optional[str] = None,
              evaluation_enabled: bool = True) -> Trainer:
        resolved = ""
        provenance_verified = True
        if resume_checkpoint:
            resolved = self._resolve_resume_checkpoint(resume_checkpoint)
            provenance_verified = validate_checkpoint_file(
                Path(resolved), self.checkpoint_provenance
            )

        # Validation must precede cache preparation and config replacement.
        # A rejected resume therefore leaves the existing run untouched.
        if evaluation_enabled:
            ensure_fid_reference(self.cfg, self.test_loader, self.device)
        self.cfg.save(os.path.join(self.run_dir, "config.json"))
        trainer = Trainer(
            algorithm=self.algorithm,
            train_loader=self.train_loader,
            cfg=self.cfg,
            run_dir=self.run_dir,
            device=self.device,
            eval_hook=self._eval_hook if evaluation_enabled else None,
        )
        trainer.checkpoint_dir = str(
            checkpoint_run_directory(Path(self.run_dir), self.checkpoint_run_number)
        )
        os.makedirs(trainer.checkpoint_dir, exist_ok=True)
        # Antigravity's trainer-owned checkpoint writer consumes this attribute
        # when serializing future payloads. Keeping it on Trainer also avoids a
        # required constructor change for older callers.
        trainer.checkpoint_provenance = self.checkpoint_provenance
        trainer.resume_provenance_verified = provenance_verified
        start_epoch = 1
        if resolved:
            start_epoch = trainer.load_checkpoint(resolved) + 1
        if start_epoch > self.cfg.epochs:
            print(
                f"[continue] Run is already complete at epoch {start_epoch - 1}; "
                f"target is {self.cfg.epochs}. Increase --epochs to extend it."
            )
            return trainer
        trainer.fit(start_epoch=start_epoch)
        return trainer

    def run_full(self, resume_checkpoint: Optional[str] = None) -> None:
        self.train(resume_checkpoint=resume_checkpoint, evaluation_enabled=True)
        # The trainer evaluates at configured epoch intervals. Do not repeat
        # the full evaluation when the final epoch has just triggered that hook.
        if self.cfg.epochs % self.cfg.evaluation.eval_frequency_epochs != 0:
            checkpoint_dir = checkpoint_run_directory(Path(self.run_dir), self.checkpoint_run_number)
            checkpoint_path = checkpoint_dir / f"{self.algorithm.name()}_epoch{self.cfg.epochs}.pt"
            self.evaluator.evaluate(
                self.sampler,
                nfe_values=self.cfg.evaluation.nfe_values,
                checkpoint_path=str(checkpoint_path)
            )

    def run_train_only(self, resume_checkpoint: Optional[str] = None) -> None:
        """Train without FID preparation, periodic evaluation, or final sampling."""
        self.train(resume_checkpoint=resume_checkpoint, evaluation_enabled=False)
