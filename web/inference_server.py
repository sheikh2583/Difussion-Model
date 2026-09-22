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
from urllib.parse import urlsplit

import torch
from torchvision.utils import save_image

from algorithms.base import BaseAlgorithm
from algorithms import ALGORITHM_REGISTRY
from config.config import ExperimentConfig
from models.backbone import build_backbone
from utils.checkpoints import load_algorithm_state
from utils.checkpoint_runs import latest_checkpoint_run_directory
from utils.gpu_lock import GpuLockError, acquire_gpu_lock


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


def training_losses(spec: ModelSpec) -> list[list[float]]:
    if not spec.metrics_path.is_file():
        return []
    latest: dict[int, float] = {}
    with spec.metrics_path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            if (
                record.get("record_type") == "train_epoch"
                and record.get("epoch") is not None
                and record.get("loss") is not None
            ):
                latest[int(record["epoch"])] = float(record["loss"])
    return [[epoch, latest[epoch]] for epoch in sorted(latest)]


def evaluation_summary(spec: ModelSpec) -> dict:
    """Summarize completed evaluation rows without touching checkpoints."""
    if not spec.metrics_path.is_file():
        return {"evaluation_count": 0, "best_fid": None, "best_fid_nfe": None}
    evaluations: list[dict] = []
    with spec.metrics_path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("record_type") == "evaluation" and record.get("fid") is not None:
                evaluations.append(record)
    if not evaluations:
        return {"evaluation_count": 0, "best_fid": None, "best_fid_nfe": None}
    best = min(evaluations, key=lambda row: float(row["fid"]))
    return {
        "evaluation_count": len(evaluations),
        "best_fid": float(best["fid"]),
        "best_fid_nfe": best.get("nfe"),
        "best_fid_epoch": best.get("epoch"),
    }


def model_catalog() -> list[dict]:
    catalog = []
    for spec in MODEL_SPECS.values():
        checkpoints = checkpoint_map(spec)
        losses = training_losses(spec)
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
                "losses": losses,
                "latest_loss": losses[-1][1] if losses else None,
                **evaluation_summary(spec),
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
        self, model_key: str, epoch: int, nfe: int, count: int, seed: int
    ) -> dict:
        if model_key not in MODEL_SPECS:
            raise ValueError(f"Unknown model: {model_key}")
        if nfe < 1 or nfe > 100:
            raise ValueError("NFE must be between 1 and 100")
        if count < 1 or count > MAX_IMAGES:
            raise ValueError(f"Image count must be between 1 and {MAX_IMAGES}")
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
                with torch.inference_mode():
                    samples = algorithm.sample(count, nfe, self.device)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                backbone_seconds = time.perf_counter() - started

                decode_started = time.perf_counter()
                images = decode_samples_for_display(samples, self.codec)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                decoder_seconds = time.perf_counter() - decode_started
                generation_seconds = backbone_seconds + decoder_seconds

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
            encoded_image = base64.b64encode(buffer.getvalue()).decode("ascii")

        return {
            "image": f"data:image/png;base64,{encoded_image}",
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


RUNTIME = InferenceRuntime()


class InferenceHandler(BaseHTTPRequestHandler):
    server_version = "DiffusionResults/2.0"

    def _send_json(self, payload: dict | list, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
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
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        request_path = urlsplit(self.path).path
        if request_path in PAGE_PATHS:
            self._send_html(PAGE_PATHS[request_path])
        elif request_path == "/api/models":
            self._send_json(
                {
                    "models": model_catalog(),
                    "device": str(RUNTIME.device),
                    "max_images": MAX_IMAGES,
                }
            )
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if self.path != "/api/generate":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > MAX_REQUEST_BYTES:
                raise ValueError("Invalid request size")
            request = json.loads(self.rfile.read(length))
            result = RUNTIME.generate(
                model_key=str(request.get("model", "")),
                epoch=int(request.get("checkpoint_epoch", 0)),
                nfe=int(request.get("nfe", 20)),
                count=int(request.get("num_images", 16)),
                seed=int(request.get("seed", 0)),
            )
            self._send_json(result)
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            self._send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except GpuLockError as error:
            self._send_json({"error": str(error)}, HTTPStatus.CONFLICT)
        except Exception as error:
            print(f"Inference error: {error}", flush=True)
            self._send_json(
                {"error": "Inference failed. Check the server console for details."},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def log_message(self, message_format: str, *args) -> None:
        print(f"[{self.log_date_time_string()}] {message_format % args}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the local inference UI.")
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
        help="Generate one image per available model at NFE 1 and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Populate MODEL_SPECS from the resolved results directory so the rest of
    # the module (checkpoint_map, training_losses, model_catalog, RUNTIME)
    # sees the correct paths without any further argument threading.
    global MODEL_SPECS
    MODEL_SPECS = build_model_specs(Path(args.results_dir).resolve())

    if args.self_test:
        for model in model_catalog():
            if not model["available"]:
                continue
            result = RUNTIME.generate(
                model["key"], model["default_epoch"], 1, 1, 0
            )
            print(
                json.dumps(
                    {key: value for key, value in result.items() if key != "image"},
                    indent=2,
                )
            )
        return

    server = ThreadingHTTPServer((args.host, args.port), InferenceHandler)
    print(f"Inference UI: http://{args.host}:{args.port}")
    print(f"Device: {RUNTIME.device}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping inference server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
