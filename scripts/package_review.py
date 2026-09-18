#!/usr/bin/env python3
"""
Rebuild thesis_review_package_minimal.zip from the current working tree.

Run after any commit (by Antigravity or Codex) to keep the zip in sync with
THESIS_REVIEW_PACKAGE.md and the changed source files.

What goes in the zip (optimised for Claude web / 3rd-party context window):
  - THESIS_REVIEW_PACKAGE.md        (the living review dossier, always included)
  - config/*.json                   (all experiment configs, incl. mf_full_v2.json)
  - docs/*.md                       (text notes, no binaries)
  - docs/report/                    (full LaTeX source, no PDFs)
  - results/aggregate/combined_metrics.jsonl
  - results/aggregate/summary.csv
  - results/*/config.json           (per-run config snapshots)
  - results/*/metrics/*.jsonl       (per-run JSONL logs)

What is intentionally excluded (too large or binary):
  - *.pt, *.npz, *.zip checkpoints
  - *.png, *.gif, *.pdf, *.pptx images and slides
  - docs/assets/
  - venv/, __pycache__/, .git/
  - results/*/checkpoints/
  - results/*/samples/

Usage:
    venv\\Scripts\\python.exe scripts\\package_review.py
    venv/bin/python scripts/package_review.py

Output: thesis_review_package_minimal.zip (overwrites the previous version).
No model training, GPU evaluation, or sampling is performed.
"""
from __future__ import annotations

import json
import os
import zipfile
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ZIP = PROJECT_ROOT / "thesis_review_package_minimal.zip"

# ---------------------------------------------------------------------------
# Inclusion rules
# ---------------------------------------------------------------------------

#: Paths relative to PROJECT_ROOT that are always included verbatim.
ALWAYS_INCLUDE: list[str] = [
    "THESIS_REVIEW_PACKAGE.md",
]

#: Glob patterns (relative to PROJECT_ROOT) to include.
GLOB_INCLUDE: list[tuple[str, str]] = [
    # All JSON configs
    ("config", "*.json"),
    # Markdown notes under docs/ (flat, no subdirs)
    ("docs", "*.md"),
    # LaTeX report source (no binaries)
    ("docs/report", "*.tex"),
    ("docs/report", "*.sty"),
    ("docs/report", "*.cls"),
    ("docs/report", "*.bib"),
    ("docs/report/chapters", "*.tex"),
    # Aggregate results (text only)
    ("results/aggregate", "combined_metrics.jsonl"),
    ("results/aggregate", "summary.csv"),
]

#: Glob patterns to include from every results/<run>/ directory.
PER_RUN_GLOBS: list[str] = [
    "config.json",
    "metrics/*.jsonl",
]

#: Directory names to skip entirely.
SKIP_DIRS: frozenset[str] = frozenset({
    "venv", "__pycache__", ".git", "checkpoints", "samples",
    "exports", "archive", "assets",
})

#: File suffixes to always exclude (binary / too large).
SKIP_SUFFIXES: frozenset[str] = frozenset({
    ".pt", ".npz", ".png", ".gif", ".jpg", ".jpeg",
    ".pdf", ".pptx", ".ppt", ".mp4", ".webm",
})


def _should_skip(path: Path) -> bool:
    for part in path.parts:
        if part in SKIP_DIRS:
            return True
    return path.suffix.lower() in SKIP_SUFFIXES


def _collect_files(root: Path) -> dict[str, Path]:
    """Return {archive_name: absolute_path} for every file to include."""
    files: dict[str, Path] = {}

    def _add(abs_path: Path) -> None:
        if not abs_path.is_file():
            return
        rel = abs_path.relative_to(root)
        if _should_skip(rel):
            return
        files[str(rel).replace("\\", "/")] = abs_path

    # Always-include list
    for name in ALWAYS_INCLUDE:
        _add(root / name)

    # Glob-based inclusions
    for directory, pattern in GLOB_INCLUDE:
        for match in sorted((root / directory).glob(pattern)):
            _add(match)

    # Per-run results directories
    results_dir = root / "results"
    if results_dir.is_dir():
        for run_dir in sorted(results_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            if run_dir.name in SKIP_DIRS or run_dir.name.startswith("."):
                continue
            for glob_pat in PER_RUN_GLOBS:
                for match in sorted(run_dir.glob(glob_pat)):
                    _add(match)

    return files


def build_zip(output: Path = OUTPUT_ZIP) -> Path:
    root = PROJECT_ROOT
    files = _collect_files(root)

    manifest = {
        "built_utc": datetime.now(timezone.utc).isoformat(),
        "file_count": len(files),
        "purpose": (
            "Minimal Claude-web review package. Contains THESIS_REVIEW_PACKAGE.md, "
            "all config JSONs, LaTeX report source, and JSONL metrics. "
            "Binary and large files are excluded."
        ),
    }

    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.unlink(missing_ok=True)

    try:
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("_review_manifest.json", json.dumps(manifest, indent=2))
            for arcname in sorted(files):
                zf.write(files[arcname], arcname=arcname)
        os.replace(tmp, output)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise

    size_kb = output.stat().st_size / 1024
    print(f"Review zip rebuilt: {output.name}  ({size_kb:.0f} KB, {len(files)} files)")
    return output


if __name__ == "__main__":
    build_zip()
