"""
Generic training engine. Contains ZERO algorithm-specific mathematics.
Depends only on BaseAlgorithm.training_step(batch) -> {"loss": tensor}.

Handles: epoch/batch loop, optimizer, scheduler, AMP, checkpointing,
logging, reproducible seeding, timing, and calling evaluation hooks.
"""
import itertools
import os
from typing import Callable, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from algorithms.base import BaseAlgorithm
from config.config import ExperimentConfig
from models.backbone import count_parameters
from utils.logging import setup_logger, JsonlLogger
from utils.results import ResultRecord, ResultsWriter
from utils.seed import set_seed, set_deterministic
from utils.timing import timer, peak_gpu_memory_mb, reset_peak_gpu_memory


def build_optimizer(modules: List[nn.Module], cfg: ExperimentConfig) -> torch.optim.Optimizer:
    params = itertools.chain.from_iterable(m.parameters() for m in modules)
    if cfg.optim.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=cfg.optim.learning_rate,
                                  weight_decay=cfg.optim.weight_decay)
    if cfg.optim.optimizer == "adam":
        return torch.optim.Adam(params, lr=cfg.optim.learning_rate,
                                 weight_decay=cfg.optim.weight_decay)
    if cfg.optim.optimizer == "sgd":
        return torch.optim.SGD(params, lr=cfg.optim.learning_rate,
                                weight_decay=cfg.optim.weight_decay)
    raise ValueError(f"Unknown optimizer: {cfg.optim.optimizer}")


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: ExperimentConfig):
    name = cfg.optim.scheduler
    if name == "none":
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs, **cfg.optim.scheduler_kwargs)
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, **cfg.optim.scheduler_kwargs)
    raise ValueError(f"Unknown scheduler: {name}")


class Trainer:
    def __init__(
        self,
        algorithm: BaseAlgorithm,
        train_loader: DataLoader,
        cfg: ExperimentConfig,
        run_dir: str,
        device: torch.device,
        eval_hook: Optional[Callable[[int], None]] = None,
    ):
        self.algorithm = algorithm
        self.model = algorithm.model
        self.trainable_modules = algorithm.trainable_modules()
        self.train_loader = train_loader
        self.cfg = cfg
        self.run_dir = run_dir
        self.eval_hook = eval_hook

        self.device = device
        for module in self.trainable_modules:
            module.to(self.device)

        self.optimizer = build_optimizer(self.trainable_modules, cfg)
        self.scheduler = build_scheduler(self.optimizer, cfg)
        self.scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and self.device.type == "cuda")

        self.logger = setup_logger(f"trainer.{algorithm.name()}",
                                    os.path.join(run_dir, "logs"))
        self.event_log = JsonlLogger(os.path.join(run_dir, "logs", "events.jsonl"))
        self.results = ResultsWriter(run_dir, cfg.experiment_name)

        self.checkpoint_dir = os.path.join(run_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        total_params, trainable_params = count_parameters(self.model)
        self.total_params = total_params
        self.trainable_params = trainable_params
        extra_modules = self.trainable_modules[1:]
        self.extra_params = sum(
            p.numel() for m in extra_modules for p in m.parameters()
        ) if extra_modules else None
        self.optimization_steps = 0
        self.samples_seen = 0
        self.cumulative_training_time = 0.0

    def _run_epoch(self, epoch: int) -> float:
        for module in self.trainable_modules:
            module.train()
        running_loss = 0.0
        n_batches = 0

        for batch, _ in self.train_loader:
            batch = batch.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type,
                                 enabled=self.cfg.amp and self.device.type == "cuda"):
                out = self.algorithm.training_step(batch)
                loss = out["loss"]

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item()
            n_batches += 1
            self.optimization_steps += 1
            self.samples_seen += batch.shape[0]

        if self.scheduler is not None:
            self.scheduler.step()

        return running_loss / max(n_batches, 1)

    def fit(self) -> None:
        set_seed(self.cfg.seed)
        set_deterministic(True)

        total_start = None
        with timer(self.device) as total_timer:
            for epoch in range(1, self.cfg.epochs + 1):
                reset_peak_gpu_memory(self.device)
                with timer(self.device) as epoch_timer:
                    avg_loss = self._run_epoch(epoch)

                peak_mem = peak_gpu_memory_mb(self.device)
                self.cumulative_training_time += epoch_timer["elapsed"]
                self.logger.info(
                    f"epoch={epoch} loss={avg_loss:.6f} "
                    f"time={epoch_timer['elapsed']:.3f}s peak_mem={peak_mem:.1f}MB"
                )
                self.event_log.log({
                    "event": "epoch_end", "epoch": epoch, "loss": avg_loss,
                    "time_per_epoch": epoch_timer["elapsed"], "peak_gpu_memory": peak_mem,
                })
                self.results.write(ResultRecord(
                    algorithm=self.algorithm.name(),
                    seed=self.cfg.seed,
                    record_type="train_epoch",
                    epoch=epoch,
                    loss=avg_loss,
                    # Cumulative wall-clock training time through this epoch
                    # (equal to total training time once the run finishes).
                    # Measured, not inferred from epoch/step counts.
                    training_time=self.cumulative_training_time,
                    time_per_epoch=epoch_timer["elapsed"],
                    optimization_steps=self.optimization_steps,
                    samples_seen=self.samples_seen,
                    parameter_count=self.total_params,
                    trainable_parameter_count=self.trainable_params,
                    algorithm_extra_parameter_count=self.extra_params,
                    peak_gpu_memory=peak_mem,
                ))

                if epoch % self.cfg.checkpoint_frequency_epochs == 0 or epoch == self.cfg.epochs:
                    self.save_checkpoint(epoch)

                if self.eval_hook is not None and \
                        epoch % self.cfg.evaluation.eval_frequency_epochs == 0:
                    self.eval_hook(epoch)

        total_time = total_timer["elapsed"]
        self.logger.info(f"training complete: total_time={total_time:.3f}s")
        self.event_log.log({"event": "training_end", "total_training_time": total_time})

    def save_checkpoint(self, epoch: int) -> str:
        path = os.path.join(self.checkpoint_dir, f"{self.algorithm.name()}_epoch{epoch}.pt")
        torch.save({
            "epoch": epoch,
            "seed": self.cfg.seed,
            "algorithm": self.algorithm.name(),
            "module_state_dicts": [m.state_dict() for m in self.trainable_modules],
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
            "config": self.cfg,
        }, path)
        self.logger.info(f"saved checkpoint: {path}")
        return path

    def load_checkpoint(self, path: str) -> int:
        # PyTorch >=2.6 defaults torch.load to weights_only=True, which refuses
        # to unpickle the custom ExperimentConfig object stored in our checkpoints.
        # Safe to disable here since we only ever load checkpoints this project
        # itself wrote.
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        for module, state_dict in zip(self.trainable_modules, ckpt["module_state_dicts"]):
            module.load_state_dict(state_dict)
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if self.scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        return ckpt["epoch"]
