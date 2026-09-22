#!/usr/bin/env python3
"""Move central transcripts into explicit pixel/latent/codec directories."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from annotate_training_log_spaces import classify_name


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = {"pixel", "latent", "codec"}


def sha256_file(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sidecar(log: Path) -> Path:
    return log.with_suffix(log.suffix + ".meta.json")


def metadata_for(log: Path) -> dict[str, Any]:
    sidecar = _sidecar(log)
    if sidecar.is_file():
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Sidecar is not an object: {sidecar}")
        return payload
    return classify_name(log)


def destination_for(log: Path, root: Path, fallback_device: str) -> tuple[Path, dict[str, Any]]:
    relative = log.relative_to(root)
    metadata = metadata_for(log)
    representation = metadata.get("representation_space")
    if representation not in CATEGORIES:
        raise ValueError(f"Cannot assign pixel/latent/codec representation: {log}")
    parts = relative.parts
    device = parts[0] if len(parts) > 1 and parts[0] not in CATEGORIES else fallback_device
    name = log.name
    dataset = str(metadata.get("dataset") or "unknown")
    if representation == "pixel" and not name.startswith(f"{dataset}_pixel_"):
        prefix = f"{dataset}_"
        if name.startswith(prefix):
            name = f"{dataset}_pixel_{name[len(prefix):]}"
        else:
            name = f"{dataset}_pixel_{name}"
    return root / device / representation / name, metadata


def build_plan(root: Path) -> list[dict[str, Any]]:
    device_dirs = sorted(
        path.name for path in root.iterdir()
        if path.is_dir() and path.name not in CATEGORIES
    ) if root.is_dir() else []
    fallback = device_dirs[0] if len(device_dirs) == 1 else "unknown-device"
    project_root = root.parent
    logs = sorted(path for path in root.rglob("*.log") if path.is_file())
    codec_roots = (
        project_root / "results" / "codecs",
        project_root / "results" / "scratch_vae",
    )
    logs.extend(sorted(
        path for codec_root in codec_roots if codec_root.is_dir()
        for path in codec_root.rglob("*.log") if path.is_file()
    ))
    plan: list[dict[str, Any]] = []
    destinations: set[Path] = set()
    for source in logs:
        if source.is_relative_to(root):
            destination, metadata = destination_for(source, root, fallback)
        else:
            metadata = metadata_for(source)
            if metadata.get("representation_space") != "codec":
                raise ValueError(f"Expected a codec transcript: {source}")
            algorithm = metadata.get("algorithm") or (
                "pretrained_vq_f4" if "results/codecs/" in source.as_posix()
                else "scratch_kl_vae"
            )
            destination = (
                root / fallback / "codec" /
                f"celeba_codec_{algorithm}_{source.name}"
            )
        if source == destination:
            continue
        if destination.exists() or destination in destinations:
            raise FileExistsError(f"Log migration collision: {destination}")
        destinations.add(destination)
        sidecar_source = _sidecar(source)
        sidecar_destination = _sidecar(destination)
        if sidecar_source.is_file() and sidecar_destination.exists():
            raise FileExistsError(f"Sidecar migration collision: {sidecar_destination}")
        plan.append({
            "source": source,
            "destination": destination,
            "sidecar_source": sidecar_source if sidecar_source.is_file() else None,
            "sidecar_destination": sidecar_destination,
            "metadata": metadata,
            "pre_sha256": sha256_file(source),
        })
    return plan


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.",
        suffix=".tmp", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def apply_plan(plan: list[dict[str, Any]], project_root: Path, root: Path) -> Path:
    moved_at = datetime.now(timezone.utc).isoformat()
    records: list[dict[str, Any]] = []
    for item in plan:
        source: Path = item["source"]
        destination: Path = item["destination"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
        post_digest = sha256_file(destination)
        if post_digest != item["pre_sha256"]:
            raise OSError(f"Digest changed while moving transcript: {source}")
        sidecar_source = item["sidecar_source"]
        if sidecar_source is not None:
            sidecar_destination: Path = item["sidecar_destination"]
            payload = json.loads(sidecar_source.read_text(encoding="utf-8"))
            new_relative = destination.relative_to(project_root).as_posix()
            for field in ("path", "log_path_at_annotation"):
                if field in payload:
                    payload[field] = new_relative
            _atomic_json(sidecar_destination, payload)
            sidecar_source.unlink()
        metadata = item["metadata"]
        records.append({
            "old_path": source.relative_to(project_root).as_posix(),
            "new_path": destination.relative_to(project_root).as_posix(),
            "pre_sha256": item["pre_sha256"],
            "post_sha256": post_digest,
            "representation": metadata.get("representation_space"),
            "dataset": metadata.get("dataset"),
            "algorithm": metadata.get("algorithm"),
            "move_utc": moved_at,
        })
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest = root / f"log_migration_manifest_{timestamp}.json"
    _atomic_json(manifest, {"schema_version": 1, "moves": records})
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    root = project_root / "training_logs"
    plan = build_plan(root)
    for item in plan:
        print(
            f"[{'MOVE' if args.apply else 'PLAN'}] "
            f"{item['source'].relative_to(project_root)} -> "
            f"{item['destination'].relative_to(project_root)}"
        )
    if args.apply:
        manifest = apply_plan(plan, project_root, root)
        print(f"Manifest: {manifest.relative_to(project_root)}")
    else:
        print(f"Planned moves: {len(plan)}; no files changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
