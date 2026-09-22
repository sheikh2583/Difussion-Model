from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

import scripts.package_thesis_context as context_package
from scripts.catalog_training_logs import build_catalog, parse_log, write_catalog
from scripts.annotate_training_log_spaces import annotate, classify_name
from scripts.package_thesis_context import training_log_files


def test_scratch_codec_log_metrics_are_cataloged(tmp_path: Path) -> None:
    log = tmp_path / "results" / "scratch_vae" / "logs" / "train.log"
    log.parent.mkdir(parents=True)
    log.write_text(
        "[run] training_type=scratch_codec\n"
        "[run] dataset=celeba\n"
        "[run] algorithm=scratch_kl_vae\n"
        "[run] started_utc=2026-09-21T12:00:00Z\n"
        "epoch=60/60 beta=0.00010000 loss=0.002018\n"
        "rFID < 5:             14.2381\n"
        "PSNR > 30 dB:         33.7268 dB\n"
        "minimum latent std > .1: 0.906543\n"
        "GATE RESULT: FAIL\n"
        "[run] finished_utc=2026-09-21T13:00:00Z\n"
        "[run] exit_status=2\n",
        encoding="utf-8",
    )

    record = parse_log(log.resolve(), tmp_path.resolve())

    assert record["training_type"] == "scratch_codec"
    assert record["representation_space"] == "codec"
    assert record["dataset"] == "celeba"
    assert record["algorithm"] == "scratch_kl_vae"
    assert record["completed_epoch"] == 60
    assert record["last_loss"] == 0.002018
    assert record["rfid"] == 14.2381
    assert record["psnr"] == 33.7268
    assert record["minimum_latent_std"] == 0.906543
    assert record["gate_result"] == "FAIL"
    assert record["exit_status"] == "2"


def test_unknown_logs_and_sidecar_metadata_are_preserved(tmp_path: Path) -> None:
    central = tmp_path / "training_logs" / "new_hardware" / "novel.log"
    central.parent.mkdir(parents=True)
    central.write_text("a future trainer with no known format\n", encoding="utf-8")
    run_local = tmp_path / "results" / "encoder_b" / "logs" / "run.log"
    run_local.parent.mkdir(parents=True)
    run_local.write_text("custom output\n", encoding="utf-8")
    run_local.with_suffix(".log.meta.json").write_text(
        json.dumps({"training_type": "vector_quantizer", "dataset": "celeba"}),
        encoding="utf-8",
    )

    catalog = build_catalog(tmp_path.resolve(), (tmp_path / "results").resolve())

    assert len(catalog) == 2
    by_path = {row["path"]: row for row in catalog}
    assert by_path["training_logs/new_hardware/novel.log"]["training_type"] == "unclassified"
    assert by_path["results/encoder_b/logs/run.log"]["training_type"] == "vector_quantizer"


def test_historical_log_sidecar_makes_pixel_and_latent_explicit(tmp_path: Path) -> None:
    pixel = tmp_path / "training_logs/gpu/celeba_fm_host_20260919T000000Z.log"
    latent = tmp_path / "training_logs/gpu/celeba_latent_fm_host_20260922T000000Z.log"
    pixel.parent.mkdir(parents=True)
    pixel.write_text("[run] dataset=celeba algorithm=fm\nepoch=2 loss=1.0\n")
    latent.write_text("epoch=3 loss=0.9\n")

    assert classify_name(pixel)["representation_space"] == "pixel"
    assert classify_name(latent)["representation_space"] == "latent"
    assert annotate(pixel, tmp_path, apply=True)
    assert annotate(latent, tmp_path, apply=True)

    pixel_record = parse_log(pixel.resolve(), tmp_path.resolve())
    latent_record = parse_log(latent.resolve(), tmp_path.resolve())
    assert pixel_record["dataset"] == "celeba"
    assert pixel_record["representation_space"] == "pixel"
    assert latent_record["dataset"] == "celeba_latent"
    assert latent_record["representation_space"] == "latent"


def test_run_local_and_orchestration_logs_are_classified() -> None:
    latent = Path("results/fm_celeba_latent/logs/trainer.FlowMatchingAlgorithm.log")
    pixel = Path("results/fm_cifar10/logs/trainer.FlowMatchingAlgorithm.log")
    codec = Path("results/scratch_vae/logs/train.log")
    suite = Path("results/tournament_run_20260921_000000_pid1.log")

    assert classify_name(latent)["representation_space"] == "latent"
    assert classify_name(pixel)["representation_space"] == "pixel"
    assert classify_name(codec)["representation_space"] == "codec"
    assert classify_name(suite)["representation_space"] == "pixel"
    assert classify_name(suite)["algorithm"] == "multiple"


def test_annotation_adds_uniform_transcript_identity(tmp_path: Path) -> None:
    log = tmp_path / "training_logs/gpu/pixel/cifar10_training.log"
    log.parent.mkdir(parents=True)
    log.write_text("historical suite\n", encoding="utf-8")
    identification = {
        "machine_label": "linux-lab-test-gpu-24gb",
        "gpu_name": "Test GPU",
        "gpu_memory_gb": 24,
    }

    assert annotate(log, tmp_path, apply=True, identification=identification)
    metadata = json.loads(
        log.with_suffix(".log.meta.json").read_text(encoding="utf-8")
    )
    assert metadata["annotation_schema_version"] == 2
    assert metadata["algorithm"] == "multiple"
    assert metadata["machine_label"] == "linux-lab-test-gpu-24gb"
    assert metadata["gpu_name"] == "Test GPU"
    assert metadata["gpu_memory_gb"] == 24
    assert metadata["transcript_bytes"] == len(b"historical suite\n")
    assert len(metadata["transcript_sha256"]) == 64


def test_catalog_outputs_and_context_manifest_inventory(tmp_path: Path) -> None:
    log = tmp_path / "results" / "run" / "logs" / "train.log"
    log.parent.mkdir(parents=True)
    log.write_text("epoch=1/2 loss=0.5\n", encoding="utf-8")
    catalog = build_catalog(tmp_path.resolve(), (tmp_path / "results").resolve())
    output = tmp_path / "results" / "aggregate"

    write_catalog(catalog, output)

    assert (output / "training_log_catalog.json").is_file()
    assert (output / "training_log_catalog.csv").is_file()
    assert "results/run/logs/train.log" in (
        output / "TRAINING_LOG_INDEX.md"
    ).read_text(encoding="utf-8")
    assert training_log_files([log], tmp_path) == ["results/run/logs/train.log"]


def test_context_required_inventory_covers_current_comparison_and_demo() -> None:
    required = set(context_package.REQUIRED_SOURCE_PATHS)
    assert "docs/COMPARISON_PROTOCOL.md" in required
    assert "scripts/aggregate_results.py" in required
    assert "web/inference_server.py" in required
    assert "web/inference_ui.html" in required
    assert "tests/test_web_catalog.py" in required
    assert "scripts/linux/refresh_thesis_context.sh" in required

    ignored_context = set(context_package.EXPLICIT_IGNORED_CONTEXT_PATHS)
    assert "AGENTS.md" in ignored_context
    assert "CROSS_TRACK.md" in ignored_context
    assert "BLOCKER_DECISIONS.md" in ignored_context

    aggregate = set(context_package.REQUIRED_AGGREGATE_PATHS)
    assert "results/aggregate/algorithm_progression_vs_fm.csv" in aggregate
    assert "results/aggregate/pixel_vs_latent.csv" in aggregate
    assert "results/aggregate/comparison_manifest.json" in aggregate


def test_archive_verification_checks_member_digest(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(context_package, "REQUIRED_SOURCE_PATHS", ("source.py",))
    monkeypatch.setattr(context_package, "REQUIRED_AGGREGATE_PATHS", ())
    monkeypatch.setattr(context_package, "EXPLICIT_IGNORED_CONTEXT_PATHS", ())
    archive_path = tmp_path / "context.zip"
    payload = b"print('verified')\n"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("source.py", payload)
    manifest = {
        "files": [{
            "path": "source.py",
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }]
    }

    context_package.verify_archive(archive_path, manifest)

    manifest["files"][0]["sha256"] = "0" * 64
    with pytest.raises(OSError, match="digest mismatch"):
        context_package.verify_archive(archive_path, manifest)
