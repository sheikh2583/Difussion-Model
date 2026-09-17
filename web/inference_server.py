"""Local HTTP inference service for the CIFAR-10 generative models.

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


PROJECT_ROOT = Path(__file__).resolve().parent
PAGE_PATHS = {
    "/": PROJECT_ROOT / "index.html",
    "/index.html": PROJECT_ROOT / "index.html",
    "/train": PROJECT_ROOT / "thesis_dashboard.html",
    "/thesis_dashboard.html": PROJECT_ROOT / "thesis_dashboard.html",
    "/infer": PROJECT_ROOT / "inference_ui.html",
    "/inference_ui.html": PROJECT_ROOT / "inference_ui.html",
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

        algo_cls, ckpt_prefix = _infer_algo_cls_from_ckpt_dir(ckpt_dir)
        if algo_cls is None:
            continue  # no recognised .pt files yet

        # Use the run directory name as the unique key (e.g. "fm_cifar10").
        key = run_dir.name
        with cfg_path.open(encoding="utf-8") as handle:
            raw_config = json.load(handle)
        dataset = raw_config.get("dataset", {}).get("name", "unknown dataset")
        base_label = _ALGO_LABELS.get(algo_cls.__name__, algo_cls.__name__)
        label = f"{base_label} ({dataset})"

        specs[key] = ModelSpec(
            key=key,
            label=label,
            algorithm_class=algo_cls,
            run_directory=run_dir,
            checkpoint_prefix=ckpt_prefix,
        )

    return specs


# MODEL_SPECS is populated in main() after --results-dir is parsed.
MODEL_SPECS: dict[str, ModelSpec] = {}


def checkpoint_map(spec: ModelSpec) -> dict[int, Path]:
    directory = spec.run_directory / "checkpoints"
    if not directory.is_dir():
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


def model_catalog() -> list[dict]:
    catalog = []
    for spec in MODEL_SPECS.values():
        checkpoints = checkpoint_map(spec)
        catalog.append(
            {
                "key": spec.key,
                "label": spec.label,
                "available": bool(checkpoints) and spec.config_path.is_file(),
                "epochs": list(checkpoints),
                "default_epoch": max(checkpoints) if checkpoints else None,
                "losses": training_losses(spec),
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

    def _load(self, spec: ModelSpec, epoch: int) -> tuple[BaseAlgorithm, float]:
        key = (spec.key, epoch)
        if key == self.cache_key and self.algorithm is not None:
            return self.algorithm, 0.0

        checkpoints = checkpoint_map(spec)
        if epoch not in checkpoints:
            raise ValueError(f"Checkpoint epoch {epoch} is unavailable for {spec.label}")
        if not spec.config_path.is_file():
            raise ValueError(f"Configuration is unavailable for {spec.label}")

        started = time.perf_counter()
        self.algorithm = None
        self.cache_key = None
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        config = ExperimentConfig.load(str(spec.config_path))
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

        self.algorithm = algorithm
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
            algorithm, load_seconds = self._load(spec, epoch)
            torch.manual_seed(seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
                torch.cuda.synchronize()

            started = time.perf_counter()
            with torch.inference_mode():
                images = algorithm.sample(count, nfe, self.device).cpu()
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            generation_seconds = time.perf_counter() - started

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
            "images_per_second": round(count / max(generation_seconds, 1e-9), 3),
        }


RUNTIME = InferenceRuntime()


class InferenceHandler(BaseHTTPRequestHandler):
    server_version = "CIFARInference/1.0"

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
