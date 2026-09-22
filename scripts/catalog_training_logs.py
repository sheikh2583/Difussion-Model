#!/usr/bin/env python3
"""Inventory every training transcript without assuming a fixed model list.

The catalog scans both the central ``training_logs/`` tree and run-local logs
under ``results/``. Standard ``[run] key=value`` headers and optional
``<name>.log.meta.json`` sidecars provide metadata for new training types, but
unknown logs are still retained with their path, digest, size, and timestamps.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_FIELD = re.compile(r"^\[run\]\s+([A-Za-z0-9_.-]+)=(.*)$")
STARTUP_FIELD = re.compile(r"^\[startup\]\s+([A-Za-z0-9_.-]+):\s*(.*)$")
EPOCH_LINE = re.compile(
    r"\bepoch=(\d+)(?:/(\d+))?.*?\bloss=([+\-0-9.eE]+)"
)
RFID_LINE = re.compile(r"^rFID\s*<\s*5:\s*([+\-0-9.eE]+)")
PSNR_LINE = re.compile(r"^PSNR\s*>\s*30\s*dB:\s*([+\-0-9.eE]+)")
STD_LINE = re.compile(r"^minimum latent std\s*>\s*\.1:\s*([+\-0-9.eE]+)")
GATE_LINE = re.compile(r"^GATE RESULT:\s*(\S+)")

FIELDS = [
    "path", "training_type", "representation_space", "dataset", "algorithm", "started_utc",
    "finished_utc", "exit_status", "completed_epoch", "target_epochs",
    "last_loss", "rfid", "psnr", "minimum_latent_std", "gate_result",
    "bytes", "modified_utc", "sha256",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/aggregate")
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def discover_logs(project_root: Path, results_root: Path) -> list[Path]:
    roots = [project_root / "training_logs", results_root]
    return sorted(
        {
            path.resolve()
            for root in roots
            if root.is_dir()
            for path in root.rglob("*.log")
            if path.is_file()
        }
    )


def _sidecar_metadata(log_path: Path) -> dict[str, Any]:
    candidates = [
        log_path.with_suffix(log_path.suffix + ".meta.json"),
        log_path.with_suffix(".meta.json"),
    ]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def parse_log(log_path: Path, project_root: Path) -> dict[str, Any]:
    metadata = _sidecar_metadata(log_path)
    record: dict[str, Any] = {key: metadata.get(key) for key in FIELDS}
    record["path"] = log_path.relative_to(project_root).as_posix()
    saw_epoch = False
    saw_scratch_codec = False

    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            match = RUN_FIELD.match(line)
            if match:
                key, value = match.groups()
                if key in FIELDS or key in {"kind", "name"}:
                    # Sidecars are the migration-safe authority for historical
                    # transcripts whose old headers may be incomplete.
                    if metadata.get(key) is None:
                        record[key] = value
                continue
            match = STARTUP_FIELD.match(line)
            if match:
                key, value = match.groups()
                if key in {"algorithm", "dataset"} and not record.get(key):
                    record[key] = value
                continue
            match = EPOCH_LINE.search(line)
            if match:
                completed, target, loss = match.groups()
                record["completed_epoch"] = int(completed)
                if target is not None:
                    record["target_epochs"] = int(target)
                record["last_loss"] = float(loss)
                saw_epoch = True
            elif "Scratch KL-VAE settings:" in line:
                saw_scratch_codec = True
            elif (match := RFID_LINE.match(line)):
                record["rfid"] = float(match.group(1))
            elif (match := PSNR_LINE.match(line)):
                record["psnr"] = float(match.group(1))
            elif (match := STD_LINE.match(line)):
                record["minimum_latent_std"] = float(match.group(1))
            elif (match := GATE_LINE.match(line)):
                record["gate_result"] = match.group(1).upper()

    record["training_type"] = (
        record.get("training_type")
        or record.pop("kind", None)
        or ("scratch_codec" if saw_scratch_codec else None)
        or ("model_training" if record.get("algorithm") or saw_epoch else None)
        or "unclassified"
    )
    if not record.get("representation_space"):
        dataset = str(record.get("dataset") or "")
        training_type = str(record.get("training_type") or "")
        if dataset.endswith("_latent") or training_type == "latent_diffusion":
            record["representation_space"] = "latent"
        elif dataset in {"cifar10", "celeba"} and training_type in {
            "model_training", "pixel_diffusion"
        }:
            record["representation_space"] = "pixel"
        elif "codec" in training_type:
            record["representation_space"] = "codec"
    stat = log_path.stat()
    record["bytes"] = stat.st_size
    record["modified_utc"] = datetime.fromtimestamp(
        stat.st_mtime, timezone.utc
    ).isoformat()
    record["sha256"] = _sha256(log_path)
    return {key: record.get(key) for key in FIELDS}


def build_catalog(
    project_root: Path, results_root: Path
) -> list[dict[str, Any]]:
    return [parse_log(path, project_root) for path in discover_logs(project_root, results_root)]


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.",
        suffix=".tmp", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _markdown(catalog: list[dict[str, Any]]) -> str:
    lines = [
        "# Training Log Index", "",
        (
            "Generated by `scripts/catalog_training_logs.py`. Every `.log` below "
            "is content-addressed; unfamiliar future training types remain listed as "
            "`unclassified` until they provide `[run]` metadata or a JSON sidecar."
        ),
        "",
        "| Type | Space | Dataset | Algorithm | Epoch | Loss | rFID | PSNR | Gate | Log |",
        "|---|---|---|---|---:|---:|---:|---:|---|---|",
    ]
    for row in catalog:
        values = {
            key: (
                "—" if item is None or item == ""
                else str(item).replace("|", "\\|")
            )
            for key in (
                "training_type", "representation_space", "dataset", "algorithm", "completed_epoch",
                "last_loss", "rfid", "psnr", "gate_result", "path",
            )
            for item in (row.get(key),)
        }

        lines.append(
            "| " + " | ".join([
                values["training_type"], values["representation_space"],
                values["dataset"], values["algorithm"],
                values["completed_epoch"], values["last_loss"], values["rfid"],
                values["psnr"], values["gate_result"], f"`{values['path']}`",
            ]) + " |"
        )
    return "\n".join(lines) + "\n"


def write_catalog(catalog: list[dict[str, Any]], output_dir: Path) -> None:
    json_text = json.dumps(
        {"schema_version": 2, "logs": catalog}, indent=2, sort_keys=True
    ) + "\n"
    _atomic_text(output_dir / "training_log_catalog.json", json_text)

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(catalog)
    _atomic_text(output_dir / "training_log_catalog.csv", buffer.getvalue())
    _atomic_text(output_dir / "TRAINING_LOG_INDEX.md", _markdown(catalog))


def main() -> int:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    results_root = (
        args.results_root if args.results_root.is_absolute()
        else project_root / args.results_root
    ).resolve()
    output_dir = (
        args.output_dir if args.output_dir.is_absolute()
        else project_root / args.output_dir
    ).resolve()
    catalog = build_catalog(project_root, results_root)
    print(f"Discovered training logs: {len(catalog)}")
    for record in catalog:
        print(
            f"  {record['training_type']} [{record['representation_space'] or 'unknown'}]: "
            f"{record['path']}"
        )
    if args.dry_run:
        return 0
    write_catalog(catalog, output_dir)
    print(f"Training log catalog: {output_dir / 'training_log_catalog.json'}")
    print(f"Training log index:   {output_dir / 'TRAINING_LOG_INDEX.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
