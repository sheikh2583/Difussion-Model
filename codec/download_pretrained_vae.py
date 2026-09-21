"""Operator-only downloader for the frozen pretrained AutoencoderKL.

Downloads an immutable Hugging Face revision into a local directory and writes
``source_manifest.json`` so codec validation records the resolved commit rather
than the moving ``main`` label.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_REPO = "stabilityai/sd-vae-ft-mse"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output-dir", default="./data/pretrained/sd-vae-ft-mse")
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface-hub is required; install the project requirements first"
        ) from exc

    resolved_revision = HfApi().model_info(
        args.repo_id, revision=args.revision
    ).sha
    if not resolved_revision:
        raise RuntimeError(f"Could not resolve {args.repo_id}@{args.revision}")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repo_id,
        revision=resolved_revision,
        local_dir=str(output_dir),
    )
    manifest = {
        "schema_version": 1,
        "repo_id": args.repo_id,
        "requested_revision": args.revision,
        "resolved_revision": resolved_revision,
    }
    manifest_path = output_dir / "source_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Downloaded {args.repo_id}@{resolved_revision} to {output_dir}")
    print(f"Source manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
