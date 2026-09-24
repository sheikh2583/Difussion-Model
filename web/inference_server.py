"""Local HTTP results browser and inference service for trained models.

Run from the project root:
    python web/inference_server.py

Then open http://127.0.0.1:8000 in a browser.
"""

from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import base64
import gc
import io
import json
import math
import re
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Type
from urllib.parse import parse_qs, quote, urlsplit

import torch
from torchvision.utils import save_image

from algorithms.base import BaseAlgorithm
from algorithms import ALGORITHM_REGISTRY
from config.config import ExperimentConfig
from models.backbone import build_backbone
from utils.checkpoints import load_algorithm_state
from utils.checkpoint_runs import latest_checkpoint_run_directory
from utils.gpu_lock import (
    DEFAULT_LOCK_PATH,
    GpuLockError,
    acquire_gpu_lock,
    owner_is_active,
    read_lock,
)


WEB_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = WEB_ROOT.parent
PAGE_PATHS = {
    "/": WEB_ROOT / "index.html",
    "/index.html": WEB_ROOT / "index.html",
    "/results": WEB_ROOT / "results_ui.html",
    "/results_ui.html": WEB_ROOT / "results_ui.html",
    "/infer": WEB_ROOT / "inference_ui.html",
    "/inference_ui.html": WEB_ROOT / "inference_ui.html",
}
MAX_REQUEST_BYTES = 64 * 1024
MAX_IMAGES = 64
MAX_TRAJECTORY_IMAGES = 16
MAX_TRAJECTORY_FRAMES = 12
UI_SCHEMA_VERSION = 8
CACHED_SAMPLE_NAME = re.compile(r"^epoch(\d+)_nfe(\d+)(?:_seed(\d+))?\.png$")


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    algorithm_class: Type[BaseAlgorithm]
    run_directory: Path
    checkpoint_prefix: str
    experiment_name: str
    dataset: str
    dataset_label: str
    image_size: int
    nfe_values: tuple[int, ...]
    backbone: dict
    raw_config: dict

    @property
    def config_path(self) -> Path:
        """Path to the config.json saved by ExperimentRunner alongside checkpoints."""
        return self.run_directory / "config.json"

    @property
    def metrics_path(self) -> Path:
        """Path to the JSONL metrics log written by ResultsWriter during training."""
        return self.run_directory / "metrics" / f"{self.run_directory.name}.jsonl"


# Reverse map: checkpoint filename class-name → algorithm class
# e.g. "FlowMatchingAlgorithm" → FlowMatchingAlgorithm
_CLS_NAME_TO_ALGO: dict[str, Type[BaseAlgorithm]] = {
    cls.__name__: cls for cls in ALGORITHM_REGISTRY.values()
}

# Human-readable labels keyed by algorithm class name.
_ALGO_LABELS: dict[str, str] = {
    "FlowMatchingAlgorithm": "Flow Matching",
    "FlowMatchingLognormAlgorithm": "Flow Matching + Logit-Normal",
    "MeanFlowAlgorithm": "Mean Flow",
    "MeanFlowDistillAlgorithm": "Mean Flow Distillation",
    "ConsistencyAlgorithm": "Consistency Models",
    "ReflowAlgorithm": "Rectified Flow Reflow",
    "MockAlgorithm": "Mock (smoke test)",
}

_ALGO_SHORT_LABELS: dict[str, str] = {
    "FlowMatchingAlgorithm": "FM",
    "FlowMatchingLognormAlgorithm": "FM-LN",
    "MeanFlowAlgorithm": "Mean Flow",
    "MeanFlowDistillAlgorithm": "MF-Distill",
    "ConsistencyAlgorithm": "Consistency",
    "ReflowAlgorithm": "Reflow",
    "MockAlgorithm": "Mock",
}

_DATASET_LABELS = {"cifar10": "CIFAR-10", "celeba": "CelebA"}


@torch.inference_mode()
def decode_samples_for_display(samples, codec=None, batch_size: int = 32):
    """Convert pixel states or normalized latent states into CPU RGB images."""
    if codec is None:
        images = samples.cpu()
    else:
        images = torch.cat(
            [
                codec.decode_normalised(samples[start : start + batch_size]).cpu()
                for start in range(0, samples.shape[0], batch_size)
            ],
            dim=0,
        )
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(
            "Inference output must decode to RGB before display; "
            f"got shape {tuple(images.shape)}"
        )
    return images


def encode_image_grid(images: torch.Tensor, count: int) -> str:
    buffer = io.BytesIO()
    save_image(
        images,
        buffer,
        format="png",
        nrow=max(1, math.ceil(math.sqrt(count))),
        normalize=True,
        value_range=(-1, 1),
        padding=2,
    )
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"


def _infer_algo_cls_from_ckpt_dir(ckpt_dir: Path):
    """Return (AlgorithmClass, checkpoint_prefix) by scanning .pt filenames."""
    for pt in ckpt_dir.glob("*.pt"):
        for cls_name, cls in _CLS_NAME_TO_ALGO.items():
            if pt.name.startswith(cls_name):
                return cls, f"{cls_name}_epoch"
    return None, None


def build_model_specs(results_root: Path) -> dict[str, "ModelSpec"]:
    """Auto-discover all valid experiment runs under results_root.

    A directory is considered a valid run if it contains:
      - config.json   (written by ExperimentRunner at the start of training)
      - checkpoints/  (at least one .pt file)

    The algorithm class is inferred from the checkpoint filename prefix
    (e.g. ``FlowMatchingAlgorithm_epoch10.pt`` → FlowMatchingAlgorithm).
    No dataset names or directory names are hardcoded here.
    """
    specs: dict[str, ModelSpec] = {}
    if not results_root.is_dir():
        return specs

    for run_dir in sorted(results_root.iterdir()):
        if not run_dir.is_dir():
            continue
        cfg_path  = run_dir / "config.json"
        ckpt_dir  = run_dir / "checkpoints"
        if not cfg_path.exists() or not ckpt_dir.is_dir():
            continue

        active_ckpt_dir = latest_checkpoint_run_directory(run_dir)
        if active_ckpt_dir is None:
            continue
        algo_cls, ckpt_prefix = _infer_algo_cls_from_ckpt_dir(active_ckpt_dir)
        if algo_cls is None:
            continue  # no recognised .pt files yet

        # Use the run directory name as the unique key (e.g. "fm_cifar10").
        key = run_dir.name
        try:
            with cfg_path.open(encoding="utf-8") as handle:
                raw_config = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        dataset_config = raw_config.get("dataset", {})
        dataset = str(dataset_config.get("name", "unknown")).lower()
        dataset_label = _DATASET_LABELS.get(dataset, dataset.replace("_", " ").title())
        evaluation = raw_config.get("evaluation", {})
        nfe_values = tuple(
            sorted(
                {
                    int(value)
                    for value in evaluation.get("nfe_values", (1, 5, 20))
                    if isinstance(value, int) and 1 <= value <= 100
                }
            )
        ) or (1, 5, 20)
        base_label = _ALGO_LABELS.get(algo_cls.__name__, algo_cls.__name__)
        label = f"{base_label} · {dataset_label}"

        specs[key] = ModelSpec(
            key=key,
            label=label,
            algorithm_class=algo_cls,
            run_directory=run_dir,
            checkpoint_prefix=ckpt_prefix,
            experiment_name=str(raw_config.get("experiment_name", run_dir.name)),
            dataset=dataset,
            dataset_label=dataset_label,
            image_size=int(dataset_config.get("image_size", 0) or 0),
            nfe_values=nfe_values,
            backbone=dict(raw_config.get("backbone", {})),
            raw_config=raw_config,
        )

    return specs


# MODEL_SPECS is populated in main() after --results-dir is parsed.
MODEL_SPECS: dict[str, ModelSpec] = {}


def checkpoint_map(spec: ModelSpec) -> dict[int, Path]:
    directory = latest_checkpoint_run_directory(spec.run_directory)
    if directory is None:
        return {}
    pattern = re.compile(rf"^{re.escape(spec.checkpoint_prefix)}(\d+)\.pt$")
    checkpoints: dict[int, Path] = {}
    for path in directory.glob("*.pt"):
        match = pattern.match(path.name)
        if match:
            checkpoints[int(match.group(1))] = path
    return dict(sorted(checkpoints.items()))


def cached_sample_artifacts(spec: ModelSpec) -> list[dict]:
    """Discover provenance-validated grids without opening model weights."""
    artifact_root = spec.run_directory.parent / "checkpoint_samples" / spec.key
    if not artifact_root.is_dir():
        return []
    checkpoints = checkpoint_map(spec)
    artifacts: list[dict] = []
    for path in sorted(artifact_root.glob("*.png")):
        match = CACHED_SAMPLE_NAME.match(path.name)
        if not match:
            continue
        epoch, nfe = int(match.group(1)), int(match.group(2))
        filename_seed = int(match.group(3) or 0)
        metadata_path = path.with_suffix(".json")
        if not metadata_path.is_file():
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        checkpoint = checkpoints.get(epoch)
        if not (
            isinstance(metadata, dict)
            and metadata.get("artifact_type") == "checkpoint_sample_grid"
            and metadata.get("experiment") == spec.key
            and metadata.get("dataset") == spec.dataset
            and metadata.get("epoch") == epoch
            and metadata.get("nfe") == nfe
            and metadata.get("seed", 0) == filename_seed
            and checkpoint is not None
            and metadata.get("checkpoint") == checkpoint.name
        ):
            continue
        artifacts.append(
            {
                "epoch": epoch,
                "nfe": nfe,
                "seed": filename_seed,
                "num_images": metadata.get("num_images"),
                "checkpoint": checkpoint.name,
                "dataset": spec.dataset,
                "source": "checkpoint_cache",
                "url": (
                    f"/api/cached-sample?model={quote(spec.key)}"
                    f"&epoch={epoch}&nfe={nfe}&seed={filename_seed}"
                ),
            }
        )
    return artifacts


def metric_records(spec: ModelSpec) -> list[dict]:
    """Read valid JSONL rows, tolerating a partially-written final line."""
    if not spec.metrics_path.is_file():
        return []
    records: list[dict] = []
    with spec.metrics_path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                records.append(record)
    return records


def training_history(spec: ModelSpec, records: list[dict] | None = None) -> list[dict]:
    rows: dict[int, dict] = {}
    for record in metric_records(spec) if records is None else records:
        if (
            record.get("record_type") == "train_epoch"
            and record.get("epoch") is not None
            and record.get("loss") is not None
        ):
            epoch = int(record["epoch"])
            rows[epoch] = {
                "epoch": epoch,
                "loss": float(record["loss"]),
                "training_time": record.get("training_time"),
                "time_per_epoch": record.get("time_per_epoch"),
                "samples_seen": record.get("samples_seen"),
                "optimization_steps": record.get("optimization_steps"),
                "peak_gpu_memory_mb": record.get("peak_gpu_memory"),
                "parameter_count": record.get("parameter_count"),
                "trainable_parameter_count": record.get("trainable_parameter_count"),
                "algorithm_extra_parameter_count": record.get(
                    "algorithm_extra_parameter_count"
                ),
            }
    return [rows[epoch] for epoch in sorted(rows)]


def training_losses(spec: ModelSpec) -> list[list[float]]:
    return [[row["epoch"], row["loss"]] for row in training_history(spec)]


def _evaluation_epoch(record: dict) -> int | None:
    if record.get("epoch") is not None:
        return int(record["epoch"])
    match = re.search(r"_epoch(\d+)\.pt$", str(record.get("checkpoint_path") or ""))
    return int(match.group(1)) if match else None


def evaluation_history(spec: ModelSpec, records: list[dict] | None = None) -> list[dict]:
    """Return the latest evaluation for each checkpoint epoch and NFE."""
    latest: dict[tuple[int | None, int], dict] = {}
    for record in metric_records(spec) if records is None else records:
        if record.get("record_type") != "evaluation" or record.get("fid") is None:
            continue
        nfe = int(record.get("nfe") or 0)
        if nfe < 1:
            continue
        epoch = _evaluation_epoch(record)
        latest[(epoch, nfe)] = {
            "epoch": epoch,
            "nfe": nfe,
            "fid": float(record["fid"]),
            "is_mean": record.get("is_mean"),
            "is_std": record.get("is_std"),
            "num_generated_samples": record.get("num_generated_samples"),
            "backbone_seconds": record.get("backbone_sampling_time"),
            "decoder_seconds": record.get("decoder_time"),
        }
    return sorted(
        latest.values(),
        key=lambda row: (row["epoch"] is None, row["epoch"] or 0, row["nfe"]),
    )


def sampling_history(spec: ModelSpec, records: list[dict] | None = None) -> list[dict]:
    """Return latest measured sampler timing for each NFE."""
    latest: dict[int, dict] = {}
    for record in metric_records(spec) if records is None else records:
        if record.get("record_type") != "sampling" or record.get("nfe") is None:
            continue
        nfe = int(record["nfe"])
        latest[nfe] = {
            "nfe": nfe,
            "sampling_seconds": record.get("sampling_time"),
            "time_per_image": record.get("time_per_image"),
            "images_per_second": record.get("images_per_second"),
            "peak_gpu_memory_mb": record.get("peak_gpu_memory"),
        }
    return [latest[nfe] for nfe in sorted(latest)]


def evidence_metadata(spec: ModelSpec, records: list[dict] | None = None) -> dict:
    """Reconstruct configuration and provenance facts used by the demo."""
    records = metric_records(spec) if records is None else records
    config = spec.raw_config
    evaluation = config.get("evaluation", {})
    dataset = config.get("dataset", {})
    optimizer = config.get("optim", {})

    def distinct(field: str) -> list:
        return list(dict.fromkeys(row[field] for row in records if row.get(field) is not None))

    code_identities = distinct("code_identity")
    machine_labels = distinct("machine_label")
    session_ids = distinct("session_id")
    config_hashes = distinct("config_sha256")
    source_identities = distinct("source_identity_sha256")
    latest_command = next(
        (row.get("command") for row in reversed(records) if row.get("command")), None
    )
    source_manifest = next(
        (
            row.get("source_manifest")
            for row in reversed(records)
            if row.get("source_manifest")
        ),
        None,
    )
    checkpoint_series = latest_checkpoint_run_directory(spec.run_directory)
    fid_cache = str(evaluation.get("fid_reference_cache", ""))
    codec_checkpoint = str(dataset.get("codec_checkpoint", ""))
    backbone_key, _ = backbone_identity(spec)
    protocol_parts = (
        spec.dataset,
        spec.image_size,
        backbone_key,
        config.get("seed", 0),
        evaluation.get("num_generated_samples"),
        fid_cache,
    )
    return {
        "seed": config.get("seed", 0),
        "amp": bool(config.get("amp", True)),
        "batch_size": config.get("batch_size"),
        "configured_epochs": config.get("epochs"),
        "checkpoint_frequency_epochs": config.get("checkpoint_frequency_epochs"),
        "optimizer": optimizer.get("optimizer"),
        "learning_rate": optimizer.get("learning_rate"),
        "weight_decay": optimizer.get("weight_decay"),
        "gradient_clip_norm": optimizer.get("gradient_clip_norm"),
        "evaluation_samples": evaluation.get("num_generated_samples"),
        "evaluation_frequency_epochs": evaluation.get("eval_frequency_epochs"),
        "fid_reference_cache": Path(fid_cache).name if fid_cache else None,
        "codec_checkpoint": Path(codec_checkpoint).name if codec_checkpoint else None,
        "checkpoint_series": checkpoint_series.name if checkpoint_series else None,
        "code_identities": code_identities,
        "machine_labels": machine_labels,
        "session_count": len(session_ids),
        "config_hashes": config_hashes,
        "source_identities": source_identities,
        "latest_command": latest_command,
        "source_manifest": Path(source_manifest).name if source_manifest else None,
        "mixed_code_identity": len(code_identities) > 1,
        "mixed_config_identity": len(config_hashes) > 1,
        "protocol_key": json.dumps(protocol_parts, separators=(",", ":")),
    }


def evaluation_summary(spec: ModelSpec, records: list[dict] | None = None) -> dict:
    """Summarize completed evaluation rows without touching checkpoints."""
    evaluations = evaluation_history(spec, records)
    if not evaluations:
        return {"evaluation_count": 0, "best_fid": None, "best_fid_nfe": None}
    best = min(evaluations, key=lambda row: float(row["fid"]))
    return {
        "evaluation_count": len(evaluations),
        "best_fid": float(best["fid"]),
        "best_fid_nfe": best.get("nfe"),
        "best_fid_epoch": best.get("epoch"),
    }


def backbone_identity(spec: ModelSpec) -> tuple[str, str]:
    config = spec.backbone
    name = str(config.get("name", "unknown"))
    multipliers = "×".join(str(value) for value in config.get("channel_mults", []))
    key = ":".join(
        str(value)
        for value in (
            name,
            config.get("in_channels", "?"),
            config.get("base_channels", "?"),
            multipliers,
            config.get("num_res_blocks", "?"),
        )
    )
    label = f"{name} · C{config.get('base_channels', '?')} · [{multipliers or '?'}]"
    return key, label


def model_catalog() -> list[dict]:
    catalog = []
    for spec in MODEL_SPECS.values():
        checkpoints = checkpoint_map(spec)
        records = metric_records(spec)
        history = training_history(spec, records)
        losses = [[row["epoch"], row["loss"]] for row in history]
        evaluations = evaluation_history(spec, records)
        samplings = sampling_history(spec, records)
        cached_samples = cached_sample_artifacts(spec)
        backbone_key, backbone_label = backbone_identity(spec)
        catalog.append(
            {
                "key": spec.key,
                "label": spec.label,
                "short_label": _ALGO_SHORT_LABELS.get(
                    spec.algorithm_class.__name__, spec.algorithm_class.__name__
                ),
                "algorithm": spec.experiment_name,
                "dataset": spec.dataset,
                "representation": (
                    "latent" if spec.dataset.endswith("_latent") else "pixel"
                ),
                "dataset_label": spec.dataset_label,
                "image_size": spec.image_size,
                "available": bool(checkpoints) and spec.config_path.is_file(),
                "epochs": list(checkpoints),
                "default_epoch": max(checkpoints) if checkpoints else None,
                "nfe_values": list(spec.nfe_values),
                "backbone_key": backbone_key,
                "backbone_label": backbone_label,
                "backbone": spec.backbone,
                "losses": losses,
                "training_history": history,
                "evaluations": evaluations,
                "sampling_history": samplings,
                "cached_samples": cached_samples,
                "evidence": evidence_metadata(spec, records),
                "latest_loss": losses[-1][1] if losses else None,
                **evaluation_summary(spec, records),
            }
        )
    return catalog


class InferenceRuntime:
    """Loads one checkpoint at a time and serializes access to the GPU."""

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lock = threading.Lock()
        self.cache_key: tuple[str, int] | None = None
        self.algorithm: BaseAlgorithm | None = None
        self.codec = None
        self.codec_key: str | None = None

    def _load(self, spec: ModelSpec, epoch: int) -> tuple[BaseAlgorithm, float]:
        key = (spec.key, epoch)
        if key == self.cache_key and self.algorithm is not None:
            return self.algorithm, 0.0

        checkpoints = checkpoint_map(spec)
        if epoch not in checkpoints:
            raise ValueError(f"Checkpoint epoch {epoch} is unavailable for {spec.label}")
        if not spec.config_path.is_file():
            raise ValueError(f"Configuration is unavailable for {spec.label}")

        config = ExperimentConfig.load(str(spec.config_path))
        codec_path: Path | None = None
        if config.dataset.name.endswith("_latent"):
            if not config.dataset.codec_checkpoint:
                raise ValueError(
                    f"Latent inference for {spec.label} requires dataset.codec_checkpoint"
                )
            codec_path = Path(config.dataset.codec_checkpoint).expanduser()
            if not codec_path.is_absolute():
                codec_path = PROJECT_ROOT / codec_path
            codec_path = codec_path.resolve()

        started = time.perf_counter()
        self.algorithm = None
        self.cache_key = None
        if codec_path is None or self.codec_key != str(codec_path):
            self.codec = None
            self.codec_key = None
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        model = build_backbone(config.backbone, image_size=config.dataset.image_size)
        algorithm = spec.algorithm_class(model, algorithm_kwargs=config.algorithm_kwargs)
        for module in algorithm.trainable_modules():
            module.to(self.device)

        checkpoint = torch.load(
            checkpoints[epoch], map_location=self.device, weights_only=False
        )
        load_algorithm_state(algorithm, checkpoint)
        for module in algorithm.trainable_modules():
            module.eval()

        codec = self.codec
        if codec_path is not None and codec is None:
            from codec.codec_factory import load_codec

            codec = load_codec(str(codec_path), self.device, require_frozen=True)

        self.algorithm = algorithm
        self.codec = codec
        self.codec_key = str(codec_path) if codec_path is not None else None
        self.cache_key = key
        return algorithm, time.perf_counter() - started

    def generate(
        self,
        model_key: str,
        epoch: int,
        nfe: int,
        count: int,
        seed: int,
        include_trajectory: bool = False,
    ) -> dict:
        if model_key not in MODEL_SPECS:
            raise ValueError(f"Unknown model: {model_key}")
        if nfe < 1 or nfe > 100:
            raise ValueError("NFE must be between 1 and 100")
        if count < 1 or count > MAX_IMAGES:
            raise ValueError(f"Image count must be between 1 and {MAX_IMAGES}")
        if include_trajectory and count > MAX_TRAJECTORY_IMAGES:
            raise ValueError(
                f"Trajectory playback supports at most {MAX_TRAJECTORY_IMAGES} images"
            )
        if seed < 0 or seed > 2**32 - 1:
            raise ValueError("Seed must be between 0 and 4294967295")

        spec = MODEL_SPECS[model_key]
        with self.lock:
            with acquire_gpu_lock(
                command=f"web inference: {model_key} epoch={epoch} nfe={nfe}"
            ):
                algorithm, load_seconds = self._load(spec, epoch)
                torch.manual_seed(seed)
                if self.device.type == "cuda":
                    torch.cuda.manual_seed_all(seed)
                    torch.cuda.synchronize()

                started = time.perf_counter()
                captured_states: list[torch.Tensor] = []
                capture_handle = None
                if include_trajectory:
                    sampling_model = getattr(algorithm, "ema_model", None) or algorithm.model
                    capture_module = getattr(sampling_model, "in_conv", sampling_model)

                    def capture_input(_module, inputs):
                        if inputs and isinstance(inputs[0], torch.Tensor):
                            # Keep the small, bounded trajectory on-device while
                            # sampling so a GPU→CPU synchronization is not forced
                            # after every function evaluation.
                            captured_states.append(inputs[0].detach().clone())

                    capture_handle = capture_module.register_forward_pre_hook(capture_input)
                with torch.inference_mode():
                    try:
                        samples = algorithm.sample(count, nfe, self.device)
                    finally:
                        if capture_handle is not None:
                            capture_handle.remove()
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                backbone_seconds = time.perf_counter() - started

                decode_started = time.perf_counter()
                images = decode_samples_for_display(samples, self.codec)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                decoder_seconds = time.perf_counter() - decode_started
                generation_seconds = backbone_seconds + decoder_seconds

            encoded_image = encode_image_grid(images, count)
            trajectory: list[dict] = []
            if include_trajectory:
                captured_states.append(samples.detach())
                if captured_states:
                    frame_count = min(MAX_TRAJECTORY_FRAMES, len(captured_states))
                    indices = sorted(
                        {
                            round(index * (len(captured_states) - 1) / max(frame_count - 1, 1))
                            for index in range(frame_count)
                        }
                    )
                    for state_index in indices:
                        if state_index == len(captured_states) - 1:
                            frame_image = encoded_image
                        else:
                            display_state = captured_states[state_index]
                            if self.codec is not None:
                                display_state = display_state.to(self.device)
                            decoded = decode_samples_for_display(
                                display_state, self.codec
                            )
                            frame_image = encode_image_grid(decoded, count)
                        trajectory.append(
                            {
                                "step": min(state_index, nfe),
                                "progress": round(state_index / max(nfe, 1), 4),
                                "image": frame_image,
                            }
                        )

        return {
            "image": encoded_image,
            "trajectory": trajectory,
            "model": model_key,
            "model_label": spec.label,
            "checkpoint_epoch": epoch,
            "nfe": nfe,
            "seed": seed,
            "num_images": count,
            "device": str(self.device),
            "checkpoint_load_seconds": round(load_seconds, 4),
            "generation_seconds": round(generation_seconds, 4),
            "backbone_seconds": round(backbone_seconds, 4),
            "decoder_seconds": round(decoder_seconds, 4),
            "images_per_second": round(count / max(generation_seconds, 1e-9), 3),
        }


# The thesis studio is intentionally read-only.  Keeping the runtime class
# available preserves import compatibility for focused CPU tests, but no
# runtime is instantiated and no model/checkpoint path is callable by HTTP.
RUNTIME: InferenceRuntime | None = None


def inference_availability(lock_path: Path | str = DEFAULT_LOCK_PATH) -> dict:
    """Return sanitized, read-only inference availability for the browser."""
    requested = Path(lock_path).expanduser()
    resolved = (
        requested if requested.is_absolute() else PROJECT_ROOT / requested
    ).resolve()
    if not resolved.exists():
        return {"available": True, "status": "available"}
    try:
        payload = read_lock(resolved)
        active = owner_is_active(payload)
    except GpuLockError:
        return {
            "available": False,
            "status": "unreadable_lock",
            "message": "The project GPU lock requires operator inspection.",
        }
    return {
        "available": False,
        "status": "active" if active else "stale",
        "pid": payload.get("pid"),
        "hostname": payload.get("hostname"),
        "command": payload.get("command"),
        "acquired_utc": payload.get("acquired_utc"),
        "message": (
            "Another project workflow is using the GPU."
            if active
            else "A stale GPU lock requires explicit operator recovery."
        ),
    }


class InferenceHandler(BaseHTTPRequestHandler):
    server_version = "DiffusionResults/2.0"

    def _send_json(self, payload: dict | list, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Thesis-UI-Version", str(UI_SCHEMA_VERSION))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, page_path: Path) -> None:
        if not page_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, f"{page_path.name} is missing")
            return
        body = page_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Thesis-UI-Version", str(UI_SCHEMA_VERSION))
        self.end_headers()
        self.wfile.write(body)

    def _send_cached_sample(self, query: str) -> None:
        parameters = parse_qs(query)
        model_key = parameters.get("model", [""])[0]
        try:
            epoch = int(parameters.get("epoch", [""])[0])
            nfe = int(parameters.get("nfe", [""])[0])
            seed = int(parameters.get("seed", ["0"])[0])
        except ValueError:
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid cached-sample selection")
            return
        spec = MODEL_SPECS.get(model_key)
        if spec is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        artifact = next(
            (
                item
                for item in cached_sample_artifacts(spec)
                if item["epoch"] == epoch
                and item["nfe"] == nfe
                and item["seed"] == seed
            ),
            None,
        )
        if artifact is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        artifact_root = spec.run_directory.parent / "checkpoint_samples" / spec.key
        path = artifact_root / f"epoch{epoch:03d}_nfe{nfe}_seed{seed}.png"
        if not path.is_file() and seed == 0:
            path = artifact_root / f"epoch{epoch:03d}_nfe{nfe}.png"
        if not path.is_file() or path.resolve().parent != artifact_root.resolve():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Thesis-UI-Version", str(UI_SCHEMA_VERSION))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        request = urlsplit(self.path)
        request_path = request.path
        if request_path in PAGE_PATHS:
            self._send_html(PAGE_PATHS[request_path])
        elif request_path == "/api/models":
            self._send_json(
                {
                    "models": model_catalog(),
                    "max_images": MAX_IMAGES,
                    "mode": "read_only_reconstruction",
                    "live_inference": False,
                    "ui_schema_version": UI_SCHEMA_VERSION,
                }
            )
        elif request_path == "/api/cached-sample":
            self._send_cached_sample(request.query)
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if self.path == "/api/generate":
            self._send_json(
                {
                    "error": (
                        "Obsolete UI tab detected. This studio no longer performs "
                        "inference. Hard-refresh the page (Ctrl+Shift+R) to load "
                        "the read-only reconstruction interface."
                    ),
                    "code": "obsolete_ui_reload_required",
                    "ui_schema_version": UI_SCHEMA_VERSION,
                },
                HTTPStatus.GONE,
            )
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def log_message(self, message_format: str, *args) -> None:
        print(f"[{self.log_date_time_string()}] {message_format % args}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the read-only thesis UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--results-dir",
        default=str(PROJECT_ROOT / "results"),
        help=(
            "Root directory that contains trained model sub-directories "
            "(e.g. fm_cifar10/, mf_cifar10/).  Defaults to ./results relative "
            "to the project root.  Override when checkpoints live elsewhere, "
            "e.g. --results-dir /data/diffusion/results"
        ),
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Validate the read-only catalog and exit; no model is executed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Populate MODEL_SPECS from the resolved results directory so the catalog
    # sees the requested evidence root without further argument threading.
    global MODEL_SPECS
    MODEL_SPECS = build_model_specs(Path(args.results_dir).resolve())

    if args.self_test:
        print(json.dumps({"models": len(model_catalog()), "mode": "read_only"}))
        return

    server = ThreadingHTTPServer((args.host, args.port), InferenceHandler)
    print(f"Thesis UI: http://{args.host}:{args.port}")
    print("Mode: read-only reconstruction (no model/GPU execution)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping inference server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
