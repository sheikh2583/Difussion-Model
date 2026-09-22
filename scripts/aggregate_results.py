#!/usr/bin/env python3
"""Merge experiment JSONL metrics, build the thesis table, and render plots."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from utils.plots import (
    plot_fid_vs_sampling_time,
    plot_gpu_memory_comparison,
    plot_is_vs_nfe,
    plot_loss_vs_epoch,
    plot_sampling_time_vs_nfe,
    plot_training_time_comparison,
)


METHOD_KEYS = {
    "FlowMatchingAlgorithm": "fm",
    "FlowMatchingLognormAlgorithm": "fm_lognorm",
    "MeanFlowAlgorithm": "mf",
    "MeanFlowDistillAlgorithm": "mf_distill",
    "ConsistencyAlgorithm": "consistency",
    "ReflowAlgorithm": "reflow",
}
METHOD_ORDER = {
    "fm": 0,
    "fm_lognorm": 1,
    "mf": 2,
    "mf_distill": 3,
    "consistency": 4,
    "reflow": 5,
}


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _dataset_identity(dataset: str) -> tuple[str, str]:
    """Return the semantic dataset and representation independently."""
    normalized = dataset.lower()
    if normalized.endswith("_latent"):
        return normalized.removesuffix("_latent"), "latent"
    return normalized, "pixel"


def _config_metadata(config: dict[str, Any], dataset: str) -> dict[str, Any]:
    dataset_config = config.get("dataset", {})
    backbone = config.get("backbone", {})
    evaluation = config.get("evaluation", {})
    dataset_family, representation = _dataset_identity(dataset)
    channels = backbone.get("in_channels") if isinstance(backbone, dict) else None
    resolution = (
        dataset_config.get("image_size") if isinstance(dataset_config, dict) else None
    )
    state_values = (
        int(channels) * int(resolution) ** 2
        if channels is not None and resolution is not None
        else None
    )
    return {
        "dataset_family": dataset_family,
        "representation": representation,
        "backbone_name": backbone.get("name") if isinstance(backbone, dict) else None,
        "backbone_signature": (
            _sha256_json(backbone)[:16] if isinstance(backbone, dict) and backbone else None
        ),
        "state_channels": channels,
        "state_resolution": resolution,
        "state_values": state_values,
        "evaluation_sample_target": (
            evaluation.get("num_generated_samples")
            if isinstance(evaluation, dict)
            else None
        ),
        "evaluation_reference": (
            evaluation.get("fid_reference_cache")
            if isinstance(evaluation, dict)
            else None
        ),
        "codec_checkpoint": (
            dataset_config.get("codec_checkpoint")
            if isinstance(dataset_config, dict)
            else None
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate all experiment metrics into plots and a summary table."
    )
    parser.add_argument("--results-root", default="results")
    parser.add_argument(
        "--output-dir", default="results/aggregate",
        help="Directory for combined JSONL, CSV, and plots.",
    )
    parser.add_argument(
        "--markdown-output",
        default=str(PROJECT_ROOT / "THESIS_SUMMARY.md"),
        help="Path for the concise Markdown thesis summary.",
    )
    return parser.parse_args()


def load_records(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        run_dir = path.parent.parent
        config: dict[str, Any] = {}
        try:
            config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
        dataset_config = config.get("dataset", {})
        dataset = dataset_config.get("name") if isinstance(dataset_config, dict) else None
        if not dataset:
            dataset = next(
                (name for name in ("cifar10", "celeba") if run_dir.name.endswith(f"_{name}")),
                "unknown",
            )
        config_metadata = _config_metadata(config, str(dataset))
        file_records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON in {path}:{line_number}: {error}"
                    ) from error
                if not isinstance(record, dict):
                    raise ValueError(f"Expected an object in {path}:{line_number}")
                file_records.append(record)

        # Preserve the raw JSONL order long enough to associate periodic
        # sampling/evaluation blocks with their preceding training epoch.
        last_epoch = None
        for record in file_records:
            if record.get("record_type") == "train_epoch" and record.get("epoch"):
                last_epoch = record["epoch"]
            elif (
                record.get("record_type") in {"evaluation", "sampling"}
                and record.get("epoch") is None
            ):
                record["epoch"] = last_epoch

        # Standalone evaluations may not follow a training row. Recover their
        # epoch from the checkpoint filename when it is available.
        for record in file_records:
            if (
                record.get("record_type") == "evaluation"
                and record.get("epoch") is None
            ):
                checkpoint = record.get("checkpoint_path") or ""
                match = re.search(r"epoch(\d+)", checkpoint)
                if match:
                    record["epoch"] = int(match.group(1))

        for record in file_records:
            experiment = path.parent.parent.name
            algorithm_class = record.get("algorithm")

            # Legacy evaluation rows did not record their checkpoint. Add
            # the conventional run_1 path only when that file exists.
            if (
                record.get("record_type") == "evaluation"
                and not record.get("checkpoint_path")
                and record.get("epoch") is not None
                and algorithm_class
            ):
                expected = (
                    path.parent.parent
                    / "checkpoints"
                    / "run_1"
                    / f"{algorithm_class}_epoch{record['epoch']}.pt"
                )
                if expected.is_file():
                    record["checkpoint_path"] = str(expected.resolve())

            machine = record.get("machine_label") or record.get("hostname")
            code_identity = record.get("code_identity") or record.get("git_commit")
            comparison = experiment
            if machine:
                comparison += f"@{machine}"
            if code_identity:
                comparison += f"@{str(code_identity)[:20]}"
            record["algorithm_class"] = algorithm_class
            record["experiment"] = experiment
            record["experiment_name"] = config.get("experiment_name", experiment)
            record["dataset"] = str(dataset).lower()
            record.update(config_metadata)
            method_key = METHOD_KEYS.get(
                str(algorithm_class), str(config.get("experiment_name", experiment))
            )
            record["method_key"] = method_key
            record["method_order"] = METHOD_ORDER.get(method_key)
            record["comparison"] = comparison
            # Plotting utilities group on `algorithm`; use the run name so
            # datasets, machines, and code revisions cannot merge silently.
            record["algorithm"] = comparison
            records.append(record)
    return records


def deduplicate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the latest row for each logical experiment measurement."""
    latest: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in records:
        key = (
            record.get("experiment"),
            record.get("machine_label") or record.get("hostname"),
            record.get("code_identity") or record.get("git_commit"),
            record.get("seed"),
            record.get("record_type"),
            record.get("epoch"),
            record.get("nfe"),
        )
        latest[key] = record
    return sorted(
        latest.values(),
        key=lambda row: (
            str(row.get("algorithm", "")),
            str(row.get("record_type", "")),
            row.get("epoch") if row.get("epoch") is not None else -1,
            row.get("nfe") if row.get("nfe") is not None else -1,
        ),
    )


def build_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_algorithm: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        algorithm = str(record.get("algorithm", "unknown"))
        by_algorithm.setdefault(algorithm, []).append(record)

    rows: list[dict[str, Any]] = []
    for algorithm, algorithm_records in sorted(by_algorithm.items()):
        training = sorted(
            (
                row for row in algorithm_records
                if row.get("record_type") == "train_epoch"
            ),
            key=lambda row: row.get("epoch") or -1,
        )
        evaluations = {
            row.get("nfe"): row
            for row in algorithm_records
            if row.get("record_type") == "evaluation"
        }
        evaluation_rows = [
            row for row in algorithm_records
            if row.get("record_type") == "evaluation"
        ]
        sample_count = next(
            (
                row.get("num_generated_samples")
                for row in reversed(evaluation_rows)
                if row.get("num_generated_samples") is not None
            ),
            "unknown",
        )
        eval_epoch = max(
            (
                row["epoch"] for row in evaluation_rows
                if row.get("epoch") is not None
            ),
            default=None,
        )
        last_training = training[-1] if training else {}
        rows.append({
            "experiment": algorithm,
            "source_experiment": next(
                (row.get("experiment") for row in algorithm_records), None
            ),
            "dataset": next(
                (row.get("dataset") for row in algorithm_records if row.get("dataset")),
                None,
            ),
            "dataset_family": next(
                (row.get("dataset_family") for row in algorithm_records
                 if row.get("dataset_family")),
                None,
            ),
            "representation": next(
                (row.get("representation") for row in algorithm_records
                 if row.get("representation")),
                None,
            ),
            "method_key": next(
                (row.get("method_key") for row in algorithm_records
                 if row.get("method_key")),
                None,
            ),
            "method_order": next(
                (row.get("method_order") for row in algorithm_records
                 if row.get("method_order") is not None),
                None,
            ),
            "machine_label": next(
                (row.get("machine_label") or row.get("hostname")
                 for row in algorithm_records
                 if row.get("machine_label") or row.get("hostname")),
                None,
            ),
            "git_commit": next(
                (row.get("git_commit") for row in algorithm_records
                 if row.get("git_commit")),
                None,
            ),
            "code_identity": next(
                (row.get("code_identity") for row in algorithm_records
                 if row.get("code_identity")),
                None,
            ),
            "config_sha256": next(
                (row.get("config_sha256") for row in algorithm_records
                 if row.get("config_sha256")),
                None,
            ),
            "gpu_name": next(
                (row.get("gpu_name") for row in algorithm_records
                 if row.get("gpu_name")),
                None,
            ),
            "algorithm_class": next(
                (row.get("algorithm_class") for row in algorithm_records
                 if row.get("algorithm_class")),
                None,
            ),
            "backbone_name": next(
                (row.get("backbone_name") for row in algorithm_records
                 if row.get("backbone_name")),
                None,
            ),
            "backbone_signature": next(
                (row.get("backbone_signature") for row in algorithm_records
                 if row.get("backbone_signature")),
                None,
            ),
            "state_channels": next(
                (row.get("state_channels") for row in algorithm_records
                 if row.get("state_channels") is not None),
                None,
            ),
            "state_resolution": next(
                (row.get("state_resolution") for row in algorithm_records
                 if row.get("state_resolution") is not None),
                None,
            ),
            "state_values": next(
                (row.get("state_values") for row in algorithm_records
                 if row.get("state_values") is not None),
                None,
            ),
            "codec_checkpoint": next(
                (row.get("codec_checkpoint") for row in algorithm_records
                 if row.get("codec_checkpoint")),
                None,
            ),
            "last_epoch": last_training.get("epoch"),
            "backbone_params": last_training.get("parameter_count"),
            "extra_params": last_training.get("algorithm_extra_parameter_count"),
            "training_time_seconds": last_training.get("training_time"),
            "peak_gpu_memory_mb": max(
                (
                    row.get("peak_gpu_memory") or 0
                    for row in training
                ),
                default=None,
            ),
            "eval_epoch": eval_epoch,
            "num_generated_samples": sample_count,
            "fid_at_1": (evaluations.get(1) or {}).get("fid"),
            "fid_at_2": (evaluations.get(2) or {}).get("fid"),
            "fid_at_5": (evaluations.get(5) or {}).get("fid"),
            "fid_at_10": (evaluations.get(10) or {}).get("fid"),
            "fid_at_20": (evaluations.get(20) or {}).get("fid"),
            "fid_at_50": (evaluations.get(50) or {}).get("fid"),
            "fid_at_100": (evaluations.get(100) or {}).get("fid"),
            "is_at_1": (evaluations.get(1) or {}).get("is_mean"),
            "is_at_2": (evaluations.get(2) or {}).get("is_mean"),
            "is_at_5": (evaluations.get(5) or {}).get("is_mean"),
            "is_at_10": (evaluations.get(10) or {}).get("is_mean"),
            "is_at_20": (evaluations.get(20) or {}).get("is_mean"),
            "is_at_50": (evaluations.get(50) or {}).get("is_mean"),
            "is_at_100": (evaluations.get(100) or {}).get("is_mean"),
        })
    return rows


def _final_evaluation_points(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return one final-epoch evaluation point per run identity and NFE."""
    sampling: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in records:
        if row.get("record_type") != "sampling":
            continue
        key = (
            row.get("experiment"), row.get("comparison"), row.get("epoch"),
            row.get("nfe"),
        )
        sampling[key] = row

    latest: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in records:
        if row.get("record_type") != "evaluation" or row.get("fid") is None:
            continue
        key = (row.get("experiment"), row.get("comparison"), row.get("nfe"))
        previous = latest.get(key)
        epoch = row.get("epoch") if row.get("epoch") is not None else -1
        previous_epoch = (
            previous.get("epoch")
            if previous and previous.get("epoch") is not None
            else -1
        )
        if previous is None or epoch >= previous_epoch:
            latest[key] = row

    points: list[dict[str, Any]] = []
    for row in latest.values():
        timing = sampling.get(
            (
                row.get("experiment"), row.get("comparison"), row.get("epoch"),
                row.get("nfe"),
            ),
            {},
        )
        points.append({
            "experiment": row.get("experiment"),
            "comparison": row.get("comparison"),
            "dataset": row.get("dataset"),
            "dataset_family": row.get("dataset_family") or _dataset_identity(
                str(row.get("dataset", "unknown"))
            )[0],
            "representation": row.get("representation") or _dataset_identity(
                str(row.get("dataset", "unknown"))
            )[1],
            "method_key": row.get("method_key") or METHOD_KEYS.get(
                str(row.get("algorithm_class")), str(row.get("experiment_name", "unknown"))
            ),
            "method_order": row.get("method_order"),
            "algorithm_class": row.get("algorithm_class"),
            "epoch": row.get("epoch"),
            "nfe": row.get("nfe"),
            "seed": row.get("seed"),
            "num_generated_samples": row.get("num_generated_samples"),
            "fid": row.get("fid"),
            "is_mean": row.get("is_mean"),
            "is_std": row.get("is_std"),
            "cmmd": row.get("cmmd"),
            "kid": row.get("kid"),
            "precision": row.get("precision"),
            "recall": row.get("recall"),
            "density": row.get("density"),
            "coverage": row.get("coverage"),
            "dino_fid": row.get("dino_fid"),
            "irs": row.get("irs"),
            "sampling_time_seconds": timing.get("sampling_time"),
            "backbone_sampling_time_seconds": timing.get("backbone_sampling_time"),
            "decoder_time_seconds": timing.get("decoder_time"),
            "images_per_second": timing.get("images_per_second"),
            "machine_label": row.get("machine_label") or row.get("hostname"),
            "gpu_name": row.get("gpu_name"),
            "gpu_memory_gb": row.get("gpu_memory_gb"),
            "code_identity": row.get("code_identity") or row.get("git_commit"),
            "config_sha256": row.get("config_sha256"),
            "backbone_signature": row.get("backbone_signature"),
            "state_channels": row.get("state_channels"),
            "state_resolution": row.get("state_resolution"),
            "state_values": row.get("state_values"),
            "codec_checkpoint": row.get("codec_checkpoint"),
            "evaluation_reference": row.get("evaluation_reference"),
        })
    return sorted(
        points,
        key=lambda row: (
            str(row.get("dataset_family")), str(row.get("representation")),
            row.get("method_order") if row.get("method_order") is not None else 999,
            str(row.get("experiment")), row.get("nfe") or -1,
        ),
    )


def _fm_protocol_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("dataset_family"), row.get("representation"), row.get("epoch"),
        row.get("nfe"), row.get("seed"), row.get("num_generated_samples"),
        row.get("machine_label"), row.get("backbone_signature"),
    )


def build_algorithm_progression(
    points: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compare every method with a protocol-compatible FM baseline."""
    baselines = {
        _fm_protocol_key(row): row for row in points if row.get("method_key") == "fm"
    }
    rows: list[dict[str, Any]] = []
    for point in points:
        baseline = baselines.get(_fm_protocol_key(point))
        fid = point.get("fid")
        baseline_fid = baseline.get("fid") if baseline else None
        comparable = baseline is not None and baseline_fid not in (None, 0)
        provenance_match = (
            baseline is not None
            and baseline.get("code_identity") == point.get("code_identity")
        )
        rows.append({
            **point,
            "fm_experiment": baseline.get("experiment") if baseline else None,
            "fm_code_identity": baseline.get("code_identity") if baseline else None,
            "source_identity_match": provenance_match if baseline else None,
            "fm_fid": baseline_fid,
            "fid_delta_vs_fm": fid - baseline_fid if comparable else None,
            "fid_improvement_percent": (
                100.0 * (baseline_fid - fid) / baseline_fid if comparable else None
            ),
            "comparison_status": (
                "baseline" if point.get("method_key") == "fm" and comparable
                else "compatible" if comparable and provenance_match
                else "compatible_provenance_difference" if comparable
                else "no_protocol_compatible_fm"
            ),
        })
    return rows


def build_transfer_consistency(
    progression: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize whether an FM-relative gain has the same sign across datasets."""
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in progression:
        if row.get("method_key") == "fm" or not str(
            row.get("comparison_status")
        ).startswith("compatible"):
            continue
        key = (
            row.get("method_key"), row.get("representation"), row.get("epoch"),
            row.get("nfe"), row.get("seed"), row.get("num_generated_samples"),
        )
        grouped.setdefault(key, []).append(row)

    output: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items(), key=lambda item: str(item[0])):
        by_dataset = {
            str(row["dataset_family"]): row["fid_improvement_percent"] for row in rows
        }
        values = list(by_dataset.values())
        positive = sum(value > 0 for value in values)
        if len(values) < 2:
            consistency = "awaiting_second_dataset"
        elif positive == len(values):
            consistency = "consistent_improvement"
        elif positive == 0:
            consistency = "consistent_regression"
        else:
            consistency = "mixed_direction"
        output.append({
            "method_key": key[0],
            "representation": key[1],
            "epoch": key[2],
            "nfe": key[3],
            "seed": key[4],
            "num_generated_samples": key[5],
            "dataset_count": len(values),
            "datasets": ";".join(sorted(by_dataset)),
            "improvement_percent_by_dataset": json.dumps(by_dataset, sort_keys=True),
            "mean_improvement_percent": sum(values) / len(values),
            "minimum_improvement_percent": min(values),
            "positive_dataset_fraction": positive / len(values),
            "transfer_status": consistency,
        })
    return output


def build_representation_comparison(
    points: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Pair pixel and latent evaluations without treating them as one ranking."""
    grouped: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    for row in points:
        key = (
            row.get("dataset_family"), row.get("method_key"), row.get("epoch"),
            row.get("nfe"), row.get("seed"), row.get("num_generated_samples"),
        )
        grouped.setdefault(key, {})[str(row.get("representation"))] = row

    output: list[dict[str, Any]] = []
    for key, representations in sorted(grouped.items(), key=lambda item: str(item[0])):
        pixel = representations.get("pixel")
        latent = representations.get("latent")
        if latent is None:
            continue
        compatible = pixel is not None
        pixel_fid = pixel.get("fid") if pixel else None
        latent_fid = latent.get("fid")
        pixel_time = pixel.get("sampling_time_seconds") if pixel else None
        latent_time = latent.get("sampling_time_seconds")
        output.append({
            "dataset_family": key[0],
            "method_key": key[1],
            "epoch": key[2],
            "nfe": key[3],
            "seed": key[4],
            "num_generated_samples": key[5],
            "pixel_machine_label": pixel.get("machine_label") if pixel else None,
            "latent_machine_label": latent.get("machine_label"),
            "machine_label_match": (
                pixel.get("machine_label") == latent.get("machine_label")
                if pixel else None
            ),
            "pixel_gpu_name": pixel.get("gpu_name") if pixel else None,
            "latent_gpu_name": latent.get("gpu_name"),
            "gpu_name_match": (
                pixel.get("gpu_name") == latent.get("gpu_name") if pixel else None
            ),
            "pixel_code_identity": pixel.get("code_identity") if pixel else None,
            "latent_code_identity": latent.get("code_identity"),
            "source_identity_match": (
                pixel.get("code_identity") == latent.get("code_identity")
                if pixel else None
            ),
            "pixel_experiment": pixel.get("experiment") if pixel else None,
            "latent_experiment": latent.get("experiment"),
            "pixel_fid": pixel_fid,
            "latent_fid": latent_fid,
            "latent_minus_pixel_fid": (
                latent_fid - pixel_fid if compatible else None
            ),
            "pixel_sampling_time_seconds": pixel_time,
            "latent_sampling_time_seconds": latent_time,
            "sampling_speedup_pixel_over_latent": (
                pixel_time / latent_time
                if pixel_time is not None and latent_time not in (None, 0)
                else None
            ),
            "pixel_state_values": pixel.get("state_values") if pixel else None,
            "latent_state_values": latent.get("state_values"),
            "state_reduction_factor": (
                pixel.get("state_values") / latent.get("state_values")
                if pixel and latent.get("state_values") not in (None, 0)
                else None
            ),
            "codec_checkpoint": latent.get("codec_checkpoint"),
            "comparison_status": (
                "paired_same_protocol_and_provenance"
                if compatible
                and pixel.get("machine_label") == latent.get("machine_label")
                and pixel.get("code_identity") == latent.get("code_identity")
                else "paired_with_provenance_difference" if compatible
                else "missing_pixel_peer"
            ),
            "interpretation_warning": (
                "Latent FID includes codec reconstruction error; report codec quality "
                "and decoder time separately."
            ),
        })
    return output


def add_pareto_flags(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mark quality/compute Pareto points within each dataset representation."""
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in points:
        key = (
            row.get("dataset_family"), row.get("representation"), row.get("epoch"),
            row.get("seed"), row.get("num_generated_samples"),
            row.get("machine_label"), row.get("code_identity"),
        )
        groups.setdefault(key, []).append(row)

    output: list[dict[str, Any]] = []
    for rows in groups.values():
        for row in rows:
            dominated_nfe = any(
                other.get("fid") <= row.get("fid")
                and other.get("nfe") <= row.get("nfe")
                and (
                    other.get("fid") < row.get("fid")
                    or other.get("nfe") < row.get("nfe")
                )
                for other in rows if other is not row
            )
            time = row.get("sampling_time_seconds")
            timed = [other for other in rows if other.get("sampling_time_seconds") is not None]
            dominated_time = None if time is None else any(
                other.get("fid") <= row.get("fid")
                and other.get("sampling_time_seconds") <= time
                and (
                    other.get("fid") < row.get("fid")
                    or other.get("sampling_time_seconds") < time
                )
                for other in timed if other is not row
            )
            output.append({
                **row,
                "pareto_fid_vs_nfe": not dominated_nfe,
                "pareto_fid_vs_sampling_time": (
                    None if dominated_time is None else not dominated_time
                ),
            })
    return output


def build_comparison_coverage(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expose completed and missing cells in the experimental matrix."""
    return [
        {
            "dataset_family": row.get("dataset_family"),
            "dataset": row.get("dataset"),
            "representation": row.get("representation"),
            "method_key": row.get("method_key"),
            "method_order": row.get("method_order"),
            "experiment": row.get("source_experiment"),
            "train_epoch": row.get("last_epoch"),
            "eval_epoch": row.get("eval_epoch"),
            "num_generated_samples": row.get("num_generated_samples"),
            "has_training": row.get("last_epoch") is not None,
            "has_evaluation": row.get("eval_epoch") is not None,
            "representation_state": (
                f"{row.get('state_channels')}x{row.get('state_resolution')}x"
                f"{row.get('state_resolution')}"
                if row.get("state_channels") is not None
                and row.get("state_resolution") is not None
                else None
            ),
            "state_values": row.get("state_values"),
            "backbone_signature": row.get("backbone_signature"),
            "codec_checkpoint": row.get("codec_checkpoint"),
        }
        for row in summary
    ]


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_comparison_manifest(path: Path, records: list[dict[str, Any]]) -> None:
    """Describe comparison semantics and modern metrics without inventing values."""
    metric_specs = [
        ("fid", "lower", "implemented"),
        ("is_mean", "higher", "implemented"),
        ("cmmd", "lower", "recommended_not_implemented"),
        ("kid", "lower", "recommended_not_implemented"),
        ("precision", "higher", "recommended_not_implemented"),
        ("recall", "higher", "recommended_not_implemented"),
        ("density", "higher", "recommended_not_implemented"),
        ("coverage", "higher", "recommended_not_implemented"),
        ("dino_fid", "lower", "recommended_not_implemented"),
        ("irs", "higher", "experimental_not_implemented"),
    ]
    payload = {
        "schema_version": 1,
        "purpose": [
            "algorithm progression relative to Flow Matching",
            "cross-dataset consistency of FM-relative improvements",
            "pixel-versus-latent architecture comparison",
            "quality-versus-compute Pareto analysis",
        ],
        "comparison_rules": {
            "algorithm_baseline": "fm",
            "fid_improvement_percent": "100 * (fm_fid - method_fid) / fm_fid",
            "algorithm_match_fields": [
                "dataset_family", "representation", "epoch", "nfe", "seed",
                "num_generated_samples", "machine_label", "backbone_signature",
            ],
            "provenance_policy": (
                "Source identity is reported as a match flag. A mismatch does not "
                "erase a historical comparison, but it must be disclosed."
            ),
            "representation_policy": (
                "pixel and latent remain separate rankings; paired deltas require "
                "the same dataset family, method, epoch, NFE, seed, sample count, "
                "and machine"
            ),
            "latent_warning": (
                "Latent-space image metrics include codec reconstruction error. "
                "Report codec validation and decoder time separately."
            ),
        },
        "metric_inventory": [
            {
                "field": field,
                "direction": direction,
                "status": (
                    "available" if any(row.get(field) is not None for row in records)
                    else status
                ),
            }
            for field, direction, status in metric_specs
        ],
        "statistical_requirements": {
            "recommended_independent_seeds": 3,
            "uncertainty": (
                "Report per-seed values and confidence intervals before claiming "
                "consistent improvement; one seed is descriptive evidence only."
            ),
            "sample_policy": (
                "Use identical real references, generated sample counts, and seed "
                "policy within each controlled comparison."
            ),
        },
        "research_sources": [
            {
                "topic": "CMMD and limitations of FID",
                "title": "Rethinking FID: Towards a Better Evaluation Metric for Image Generation",
                "venue": "CVPR 2024",
                "url": "https://openaccess.thecvf.com/content/CVPR2024/html/Jayasumana_Rethinking_FID_Towards_a_Better_Evaluation_Metric_for_Image_Generation_CVPR_2024_paper.html",
            },
            {
                "topic": "alternative feature encoders and multidimensional evaluation",
                "title": "Exposing flaws of generative model evaluation metrics and their unfair treatment of diffusion models",
                "url": "https://openreview.net/forum?id=08zf7kTOoh",
            },
            {
                "topic": "diversity evaluation",
                "title": "Image Generation Diversity Issues and How to Tame Them",
                "url": "https://arxiv.org/abs/2411.16171",
            },
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def markdown_value(value: Any, digits: int | None = None) -> str:
    """Format a compact, pipe-safe Markdown table value."""
    if value is None or value == "":
        return "—"
    if isinstance(value, float) and digits is not None:
        return f"{value:.{digits}f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def write_markdown_summary(
    path: Path,
    summary: list[dict[str, Any]],
    training_cost_rows: list[dict[str, Any]],
    transfer_rows: list[dict[str, Any]] | None = None,
    representation_rows: list[dict[str, Any]] | None = None,
) -> None:
    """Write a stable, human-readable thesis snapshot from aggregate rows."""
    lines = [
        "# Thesis Results Summary",
        "",
        "Generated from the canonical experiment metric files by "
        "`scripts/aggregate_results.py`. Rerun the generator after active training "
        "finishes to produce the final snapshot.",
        "",
        "## Evaluation metrics",
        "",
        "| Dataset | Space | Experiment | Run identity | Algorithm | Train epoch | Eval epoch | Samples | "
        "FID@1 | FID@5 | FID@10 | FID@20 | FID@50 | IS@20 |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        values = [
            markdown_value(row.get("dataset")),
            markdown_value(row.get("representation")),
            markdown_value(row.get("source_experiment")),
            markdown_value(
                f"{row.get('machine_label') or 'unknown'}@"
                f"{str(row.get('code_identity') or row.get('git_commit') or 'unknown')[:12]}"
            ),
            markdown_value(row.get("algorithm_class")),
            markdown_value(row.get("last_epoch")),
            markdown_value(row.get("eval_epoch")),
            markdown_value(row.get("num_generated_samples")),
            markdown_value(row.get("fid_at_1"), 3),
            markdown_value(row.get("fid_at_5"), 3),
            markdown_value(row.get("fid_at_10"), 3),
            markdown_value(row.get("fid_at_20"), 3),
            markdown_value(row.get("fid_at_50"), 3),
            markdown_value(row.get("is_at_20"), 3),
        ]
        lines.append("| " + " | ".join(values) + " |")

    lines.extend([
        "",
        "## Controlled comparison artifacts",
        "",
        "The aggregate directory contains protocol-matched FM-relative gains, "
        "cross-dataset transfer checks, pixel-versus-latent pairs, Pareto flags, "
        "and an experiment-coverage matrix. Positive `fid_improvement_percent` "
        "means lower FID than FM. Missing peers remain explicit instead of being "
        "silently compared.",
        "",
        f"- Transfer groups available: {len(transfer_rows or [])}",
        f"- Pixel/latent pairs available: {len(representation_rows or [])}",
        "- Latent FID includes codec reconstruction error; decoder time and codec "
        "quality must be reported separately.",
        "",
        "## Interactive checkpoint demo and external handoff",
        "",
        "The local inference UI can play saved checkpoints in epoch order using "
        "one selected algorithm, NFE, image count, and fixed seed. It advances "
        "the loss curve with the checkpoint, shows a conceptual noise-to-sample "
        "transition, and decodes latent outputs through the recorded codec. "
        "Backbone and decoder timings are reported separately.",
        "",
        "Build the verified Claude Web handoff with "
        "`./scripts/linux/make_thesis_context.sh`. The command rebuilds this "
        "summary, refreshes the training-log catalog, validates comparison tables "
        "and Git-tracked implementation files, then verifies every ZIP member "
        "against its SHA-256 digest.",
    ])

    lines.extend([
        "",
        "## Training cost",
        "",
        "| Experiment | Algorithm | Epochs | Training hours | Peak GPU memory (MiB) |",
        "|---|---|---:|---:|---:|",
    ])
    for row in training_cost_rows:
        values = [
            markdown_value(row.get("experiment")),
            markdown_value(row.get("algorithm")),
            markdown_value(row.get("epochs")),
            markdown_value(row.get("total_training_hours"), 2),
            markdown_value(row.get("peak_gpu_mb"), 1),
        ]
        lines.append("| " + " | ".join(values) + " |")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


ALGORITHM_LABELS = {
    "FlowMatchingAlgorithm": "FM",
    "FlowMatchingLognormAlgorithm": "FM-LN",
    "MeanFlowAlgorithm": "Mean Flow",
    "MeanFlowDistillAlgorithm": "MF-Distill",
    "ConsistencyAlgorithm": "Consistency",
    "ReflowAlgorithm": "Reflow",
}


def experiment_labels(records: list[dict[str, Any]], dataset: str) -> dict[str, str]:
    """Build stable labels from recorded algorithm classes and config metadata."""
    labels: dict[str, str] = {}
    for row in records:
        if row.get("dataset") != dataset:
            continue
        experiment = str(row.get("experiment", "unknown"))
        algorithm_class = str(row.get("algorithm_class", ""))
        base = ALGORITHM_LABELS.get(algorithm_class, str(row.get("experiment_name", experiment)))
        # Preserve meaningful variant names without exposing the redundant
        # dataset suffix already represented by the plot title.
        configured = str(row.get("experiment_name", ""))
        canonical = {"fm", "fm_lognorm", "mf", "mf_distill", "consistency", "reflow"}
        labels[experiment] = base if configured in canonical else f"{base} [{configured}]"
    return labels


def plot_primary_loss_curves(
    records: list[dict[str, Any]], dataset: str, out_path: Path
) -> None:
    """Plot one log-scale training-loss curve per discovered dataset run."""
    plt.figure()
    for experiment, label in experiment_labels(records, dataset).items():
        by_epoch: dict[int, dict[str, Any]] = {}
        for row in records:
            if (
                row.get("experiment") == experiment
                and row.get("record_type") == "train_epoch"
                and row.get("epoch") is not None
                and row.get("loss") is not None
            ):
                by_epoch[int(row["epoch"])] = row
        if not by_epoch:
            continue
        epochs = sorted(by_epoch)
        plt.plot(
            epochs,
            [by_epoch[epoch]["loss"] for epoch in epochs],
            label=label,
        )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.title(f"{dataset_label(dataset)} Training Loss Curves")
    if plt.gca().has_data():
        plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def build_training_cost_table(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build one training-cost row per canonical experiment directory."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(
            str(record.get("experiment", "unknown")), []
        ).append(record)

    rows: list[dict[str, Any]] = []
    for experiment, experiment_records in sorted(grouped.items()):
        training = sorted(
            (
                row for row in experiment_records
                if row.get("record_type") == "train_epoch"
                and row.get("epoch") is not None
            ),
            key=lambda row: row["epoch"],
        )
        if not training:
            continue
        evaluations = [
            row for row in experiment_records
            if row.get("record_type") == "evaluation"
        ]
        final_training = training[-1]
        sample_count = next(
            (
                row.get("num_generated_samples")
                for row in reversed(evaluations)
                if row.get("num_generated_samples") is not None
            ),
            "unknown",
        )
        eval_epoch = max(
            (row["epoch"] for row in evaluations if row.get("epoch") is not None),
            default=None,
        )
        training_seconds = final_training.get("training_time")
        rows.append({
            "experiment": experiment,
            "algorithm": next(
                (
                    row.get("algorithm_class") for row in experiment_records
                    if row.get("algorithm_class")
                ),
                None,
            ),
            "epochs": final_training.get("epoch"),
            "total_training_hours": (
                training_seconds / 3600 if training_seconds is not None else None
            ),
            "peak_gpu_mb": max(
                (row.get("peak_gpu_memory") or 0 for row in training),
                default=None,
            ),
            "num_generated_samples": sample_count,
            "eval_epoch": eval_epoch,
        })
    return rows


def plot_fid_subset(
    records: list[dict[str, Any]],
    experiments: dict[str, str],
    out_path: Path,
    title: str,
) -> None:
    """Render a controlled FID comparison without unrelated runs."""
    plt.figure()
    for experiment, label in experiments.items():
        matching = [
            row for row in records
            if row.get("experiment") == experiment
            and row.get("record_type") == "evaluation"
            and row.get("fid") is not None
        ]
        comparisons = sorted(
            {str(row.get("comparison") or experiment) for row in matching}
        )
        for comparison in comparisons:
            rows = sorted(
                (
                    row for row in matching
                    if str(row.get("comparison") or experiment) == comparison
                ),
                key=lambda row: row.get("nfe") or -1,
            )
            series_label = label
            if comparison != experiment:
                series_label = f"{label} [{comparison.removeprefix(experiment + '@')}]"
            plt.plot(
                [row["nfe"] for row in rows],
                [row["fid"] for row in rows],
                marker="o",
                label=series_label,
            )
    plt.xlabel("NFE")
    plt.ylabel("FID")
    plt.title(title)
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def dataset_label(dataset: str) -> str:
    return {"cifar10": "CIFAR-10", "celeba": "CelebA"}.get(
        dataset, dataset.replace("_", " ").title()
    )


def render_plots(combined_path: Path, plot_dir: Path, records: list[dict[str, Any]]) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    record_types = {row.get("record_type") for row in records}
    if "training" in record_types or "train_epoch" in record_types:
        plot_loss_vs_epoch(str(combined_path), str(plot_dir / "loss_vs_epoch.png"))
        plot_training_time_comparison(
            str(combined_path), str(plot_dir / "training_time.png")
        )
        plot_gpu_memory_comparison(
            str(combined_path), str(plot_dir / "gpu_memory.png")
        )
        for dataset in sorted({str(row.get("dataset")) for row in records} - {"unknown"}):
            plot_primary_loss_curves(
                records, dataset, plot_dir / f"loss_curves_{dataset}.png"
            )
    if "sampling" in record_types:
        plot_sampling_time_vs_nfe(
            str(combined_path), str(plot_dir / "sampling_time_vs_nfe.png")
        )
    if "evaluation" in record_types:
        for dataset in sorted({str(row.get("dataset")) for row in records} - {"unknown"}):
            labels = experiment_labels(records, dataset)
            if labels:
                plot_fid_subset(
                    records,
                    labels,
                    plot_dir / f"fid_vs_nfe_{dataset}.png",
                    f"{dataset_label(dataset)}: FID vs NFE",
                )
        plot_is_vs_nfe(str(combined_path), str(plot_dir / "is_vs_nfe.png"))
    if {"sampling", "evaluation"} <= record_types:
        plot_fid_vs_sampling_time(
            str(combined_path), str(plot_dir / "fid_vs_sampling_time.png")
        )


def main() -> int:
    args = parse_args()
    results_root = Path(args.results_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    markdown_path = Path(args.markdown_output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_path = output_dir / "combined_metrics.jsonl"

    # Select only the canonical JSONL for each run directory: the file whose
    # stem matches the run directory name.  This prevents auxiliary files
    # (e.g. smoke_lognorm.jsonl inside fm_lognorm_rtx3060/metrics/) from
    # overwriting the real run's evaluation records during deduplication.
    all_jsonl = sorted(
        path for path in results_root.glob("*/metrics/*.jsonl")
        if path.resolve() != combined_path
        and output_dir not in path.resolve().parents
    )
    inputs: list[Path] = []
    seen_run_dirs: set[Path] = set()
    for path in all_jsonl:
        run_dir = path.parent.parent
        if path.stem == run_dir.name:
            inputs.append(path)
            seen_run_dirs.add(run_dir)
    # Warn for any run directory that has metrics files but no canonical one
    for path in all_jsonl:
        run_dir = path.parent.parent
        if run_dir not in seen_run_dirs:
            print(
                f"  [WARN] {run_dir.name}: no canonical JSONL found "
                f"(stem must equal run dir name); skipped {path.name}",
                file=sys.stderr,
            )
            seen_run_dirs.add(run_dir)  # only warn once per dir
    if not inputs:
        print(f"No metric JSONL files found under {results_root}", file=sys.stderr)
        return 1

    records = deduplicate(load_records(inputs))
    if not records:
        print("Metric files contained no records.", file=sys.stderr)
        return 1

    with combined_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    summary = build_summary(records)
    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    training_cost_rows = build_training_cost_table(records)
    training_cost_path = output_dir / "training_cost_table.csv"
    if training_cost_rows:
        with training_cost_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(training_cost_rows[0])
            )
            writer.writeheader()
            writer.writerows(training_cost_rows)

    evaluation_points = _final_evaluation_points(records)
    progression_rows = build_algorithm_progression(evaluation_points)
    transfer_rows = build_transfer_consistency(progression_rows)
    representation_rows = build_representation_comparison(evaluation_points)
    pareto_rows = add_pareto_flags(evaluation_points)
    coverage_rows = build_comparison_coverage(summary)
    comparison_outputs = {
        "comparison_coverage.csv": coverage_rows,
        "algorithm_progression_vs_fm.csv": progression_rows,
        "cross_dataset_consistency.csv": transfer_rows,
        "pixel_vs_latent.csv": representation_rows,
        "quality_compute_pareto.csv": pareto_rows,
    }
    for filename, rows in comparison_outputs.items():
        write_csv_rows(output_dir / filename, rows)
    write_comparison_manifest(output_dir / "comparison_manifest.json", records)

    write_markdown_summary(
        markdown_path,
        summary,
        training_cost_rows,
        transfer_rows,
        representation_rows,
    )

    render_plots(combined_path, output_dir / "plots", records)
    print(f"Aggregated {len(records)} records from {len(inputs)} files.")
    print(f"Combined metrics: {combined_path}")
    print(f"Summary table:    {summary_path}")
    print(f"Training costs:   {training_cost_path}")
    print(f"Comparisons:      {output_dir / 'comparison_manifest.json'}")
    print(f"Markdown summary: {markdown_path}")
    print(f"Plots:            {output_dir / 'plots'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
