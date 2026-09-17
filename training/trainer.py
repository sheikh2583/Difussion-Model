"""
Generic training engine. Contains ZERO algorithm-specific mathematics.
Depends only on BaseAlgorithm.training_step(batch) -> {"loss": tensor}.

Handles: epoch/batch loop, optimizer, scheduler, AMP, checkpointing,
logging, reproducible seeding, timing, and calling evaluation hooks.

Change from original: after each epoch, calls algorithm.on_epoch_end(epoch, total)
if the algorithm defines it (used by MeanFlowAlgorithm for delta annealing).
"""
import itertools
import json
import os
import zipfile
from datetime import datetime, timezone
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
from utils.checkpoints import load_algorithm_state


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
        run_name = os.path.basename(os.path.normpath(run_dir))
        self.results = ResultsWriter(run_dir, run_name)

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
            scale_before_step = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # A decreasing scale means GradScaler detected non-finite gradients
            # and skipped optimizer.step(); dependent state such as EMA must
            # remain unchanged in that case.
            if self.scaler.get_scale() >= scale_before_step:
                self.algorithm.on_after_optimizer_step()

            running_loss += loss.item()
            n_batches += 1
            self.optimization_steps += 1
            self.samples_seen += batch.shape[0]

        if self.scheduler is not None:
            self.scheduler.step()

        avg_loss = running_loss / max(n_batches, 1)

        # Notify algorithm that an epoch completed (used for delta annealing in MF)
        if hasattr(self.algorithm, "on_epoch_end"):
            self.algorithm.on_epoch_end(epoch, self.cfg.epochs)

        return avg_loss

    def save_checkpoint(self, epoch: int) -> None:
        path = os.path.join(
            self.checkpoint_dir,
            f"{self.algorithm.name()}_epoch{epoch}.pt"
        )
        payload = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "algorithm_state": self.algorithm.checkpoint_state(),
        }
        # Save extra module states (e.g. r_embed for MF algorithms)
        extra_modules = self.trainable_modules[1:]
        for i, m in enumerate(extra_modules):
            payload[f"extra_module_{i}_state"] = m.state_dict()

        torch.save(payload, path)
        self.logger.info(f"Checkpoint saved -> {path}")
        self._zip_checkpoint(epoch, path)

    def _zip_checkpoint(self, epoch: int, ckpt_path: str) -> None:
        """
        Pack the checkpoint + config + metadata into a self-contained zip.

        Archive layout:
            <AlgoClass>_epoch<N>.zip
              ├── checkpoint.pt     (the .pt file)
              ├── config.json       (experiment config, copied from run_dir)
              └── meta.json         (epoch, algorithm, experiment, timestamp)

        The zip is written to checkpoints/archive/ alongside the .pt files.
        It uses ZIP_STORED (no compression) because .pt files are already
        compressed tensors — compressing them again wastes time.
        """
        archive_dir = os.path.join(self.checkpoint_dir, "archive")
        os.makedirs(archive_dir, exist_ok=True)

        algo_name  = self.algorithm.name()
        zip_name   = f"{algo_name}_epoch{epoch}.zip"
        zip_path   = os.path.join(archive_dir, zip_name)

        meta = {
            "epoch":      epoch,
            "algorithm":  algo_name,
            "experiment": self.cfg.experiment_name,
            "dataset":    self.cfg.dataset.name,
            "timestamp":  datetime.now(timezone.utc).isoformat(),
        }

        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
            zf.write(ckpt_path, arcname="checkpoint.pt")

            cfg_path = os.path.join(self.run_dir, "config.json")
            if os.path.exists(cfg_path):
                zf.write(cfg_path, arcname="config.json")

            zf.writestr("meta.json", json.dumps(meta, indent=2))

        self.logger.info(f"Checkpoint archive -> {zip_path}")


    def load_checkpoint(self, path: str) -> int:
        state = torch.load(path, map_location=self.device, weights_only=False)
        load_algorithm_state(self.algorithm, state)
        optimizer_state = state.get("optimizer_state", state.get("optimizer_state_dict"))
        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)
        epoch = state.get("epoch", 0)
        self.logger.info(f"Resumed from {path} at epoch {epoch}")
        return epoch

    def fit(self) -> None:
        set_seed(self.cfg.seed)
        set_deterministic(True)

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

                if self.eval_hook and epoch % self.cfg.evaluation.eval_frequency_epochs == 0:
                    self.eval_hook(epoch)
