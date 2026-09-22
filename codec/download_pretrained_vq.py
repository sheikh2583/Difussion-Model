"""Download only the frozen VQ codec from CompVis/ldm-celebahq-256."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_REPO = "CompVis/ldm-celebahq-256"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--output-dir", default="./data/pretrained/ldm-celebahq-256"
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface-hub is required; install project requirements first"
        ) from exc

    resolved = HfApi().model_info(args.repo_id, revision=args.revision).sha
    if not resolved:
        raise RuntimeError(f"Could not resolve {args.repo_id}@{args.revision}")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repo_id,
        revision=resolved,
        local_dir=str(output_dir),
        allow_patterns=["vqvae/*"],
    )
    source_dir = output_dir / "vqvae"
    if not (source_dir / "config.json").is_file():
        raise RuntimeError(f"Downloaded snapshot has no VQ config: {source_dir}")
    manifest = {
        "schema_version": 1,
        "repo_id": args.repo_id,
        "subfolder": "vqvae",
        "requested_revision": args.revision,
        "resolved_revision": resolved,
    }
    manifest_text = json.dumps(manifest, indent=2) + "\n"
    (source_dir / "source_manifest.json").write_text(
        manifest_text, encoding="utf-8"
    )
    print(f"Downloaded {args.repo_id}/vqvae@{resolved} to {source_dir}")
    print(f"Source manifest: {source_dir / 'source_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
