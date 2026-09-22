#!/usr/bin/env python3
"""Read-only structural health check for the repository.

This utility never imports the model stack, reads checkpoint tensors, writes
results, or signals processes. It is safe to run while training is active.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLATFORM_SUFFIXES = {".sh", ".ps1", ".cmd"}
# Repository-local IDE/worktree metadata is not part of this checkout's source
# layout. In particular, .kilo/worktrees may contain complete nested clones
# whose platform launchers would otherwise be reported as misplaced files.
IGNORED_TREES = {".git", ".kilo", "venv", "results", "__pycache__"}
REPORT_FILES = {
    "docs/report/main.tex",
    "docs/report/citations.bib",
    "docs/report/iutbscthesis.cls",
    "docs/report/frontmatter.sty",
    "docs/report/personnelhandler.sty",
}
MARKDOWN_LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=PROJECT_ROOT, help="Repository root to inspect."
    )
    parser.add_argument(
        "--fail-if-training",
        action="store_true",
        help="Return a failure when a project train.py process is active.",
    )
    return parser.parse_args()


def project_files(root: Path):
    for directory, names, files in os.walk(root):
        names[:] = [name for name in names if name not in IGNORED_TREES]
        base = Path(directory)
        for name in files:
            yield base / name


def platform_placement_errors(root: Path) -> list[str]:
    errors = []
    linux = root / "scripts" / "linux"
    windows = root / "scripts" / "windows"
    for path in project_files(root):
        if path.suffix not in PLATFORM_SUFFIXES:
            continue
        if path.suffix == ".sh" and linux not in path.parents:
            errors.append(f"Linux shell script outside scripts/linux: {path.relative_to(root)}")
        if path.suffix in {".ps1", ".cmd"} and windows not in path.parents:
            errors.append(f"Windows script outside scripts/windows: {path.relative_to(root)}")
    return errors


def config_errors(root: Path) -> list[str]:
    errors = []
    for path in sorted((root / "config").glob("*.json")):
        try:
            config = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"Invalid config {path.relative_to(root)}: {error}")
            continue
        experiment = str(config.get("experiment_name", "")).strip()
        dataset = config.get("dataset", {})
        dataset_name = str(dataset.get("name", "")).strip() if isinstance(dataset, dict) else ""
        if not experiment:
            errors.append(f"Missing experiment_name: {path.relative_to(root)}")
        if dataset_name not in {"cifar10", "celeba", "celeba_latent"}:
            errors.append(f"Invalid dataset name in {path.relative_to(root)}: {dataset_name!r}")
        if dataset_name and experiment.endswith(f"_{dataset_name}"):
            errors.append(
                f"experiment_name redundantly includes dataset in {path.relative_to(root)}"
            )
    return errors


def documentation_errors(root: Path) -> list[str]:
    errors = []
    for path in project_files(root):
        if path.suffix.lower() != ".md":
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for raw_target in MARKDOWN_LINK.findall(text):
            target = raw_target.strip().strip("<>").split("#", 1)[0]
            if not target or "://" in target or target.startswith(("#", "mailto:")):
                continue
            if not (path.parent / target).resolve().exists():
                errors.append(f"Broken link in {path.relative_to(root)}: {raw_target}")
    return errors


def entrypoint_errors(root: Path) -> list[str]:
    errors = []
    for path in sorted((root / "scripts" / "linux").glob("*.sh")):
        data = path.read_bytes()
        if b"\r\n" in data:
            errors.append(f"Linux script contains CRLF: {path.relative_to(root)}")
        if not data.startswith(b"#!/usr/bin/env "):
            errors.append(f"Linux script lacks env shebang: {path.relative_to(root)}")
        if not path.stat().st_mode & stat.S_IXUSR:
            errors.append(f"Linux script is not executable: {path.relative_to(root)}")
    for path in sorted((root / "scripts" / "windows").glob("*")):
        if path.suffix.lower() not in {".ps1", ".cmd"}:
            continue
        data = path.read_bytes()
        if b"\n" in data and b"\r\n" not in data:
            errors.append(f"Windows script lacks CRLF: {path.relative_to(root)}")
    return errors


def report_errors(root: Path) -> list[str]:
    errors = [f"Missing report file: {name}" for name in sorted(REPORT_FILES) if not (root / name).is_file()]
    main = root / "docs" / "report" / "main.tex"
    if main.is_file():
        text = main.read_text(encoding="utf-8", errors="replace")
        for chapter in re.findall(r"\\input\{([^}]+)\}", text):
            if not (main.parent / f"{chapter}.tex").is_file():
                errors.append(f"Missing report chapter: docs/report/{chapter}.tex")
    return errors


def ignore_policy_errors(root: Path) -> list[str]:
    probes = (
        "results/example/checkpoints/run_1/model.pt",
        "results/aggregate/plots/example.png",
        "results/exports/example.zip",
        "data/raw/example.bin",
    )
    try:
        result = subprocess.run(
            ["git", "check-ignore", *probes],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        return [f"Cannot check Git ignore policy: {error}"]
    ignored = set(result.stdout.splitlines())
    return [f"Generated artifact is not ignored: {probe}" for probe in probes if probe not in ignored]


def active_training_processes(root: Path) -> list[tuple[int, str]]:
    """Return matching processes without requiring psutil or sending signals."""
    matches: list[tuple[int, str]] = []
    proc = Path("/proc")
    if not proc.is_dir():
        return matches
    root_text = str(root.resolve())
    own_pid = os.getpid()
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == own_pid:
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
            cwd = os.readlink(entry / "cwd")
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
            continue
        trainer_pattern = (
            r"(?:^|\s|/)(?:train[^/\s]*\.(?:py|sh))(?:\s|$)"
            r"|(?:^|\s)-m\s+[A-Za-z0-9_.]*train[A-Za-z0-9_.]*(?:\s|$)"
        )
        if cwd == root_text and re.search(trainer_pattern, command):
            matches.append((int(entry.name), command))
    return sorted(matches)


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    checks = {
        "platform placement": platform_placement_errors(root),
        "configuration naming": config_errors(root),
        "documentation links": documentation_errors(root),
        "entrypoint format": entrypoint_errors(root),
        "Git ignore policy": ignore_policy_errors(root),
        "submission report": report_errors(root),
    }
    failures = 0
    for name, errors in checks.items():
        if errors:
            failures += len(errors)
            print(f"[FAIL] {name}")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"[OK] {name}")

    training = active_training_processes(root)
    if training:
        print(f"[ACTIVE] {len(training)} training-related processes")
        for pid, command in training:
            print(f"  - PID {pid}: {command}")
        if args.fail_if_training:
            failures += 1
    else:
        print("[OK] no active project trainers")

    if failures:
        print(f"Layout verification failed with {failures} issue(s).", file=sys.stderr)
        return 1
    print("Layout verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
