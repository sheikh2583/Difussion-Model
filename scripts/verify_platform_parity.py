#!/usr/bin/env python3
"""Statically verify Linux launcher coverage and public-option parity on Windows."""

from __future__ import annotations

import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LINUX_DIR = PROJECT_ROOT / "scripts" / "linux"
WINDOWS_DIR = PROJECT_ROOT / "scripts" / "windows"


def _normalise_option(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _linux_public_options(path: Path) -> set[str]:
    """Return options declared by a launcher's top-level argv case block."""
    text = path.read_text(encoding="utf-8")
    start = text.find("while [[ $# -gt 0 ]]")
    if start < 0:
        return set()
    block = text[start:]
    end = block.find("\ndone")
    if end >= 0:
        block = block[:end]
    options: set[str] = set()
    label = re.compile(
        r"^\s*((?:--[a-z][a-z0-9-]*|-h)(?:\|--[a-z][a-z0-9-]*)*)\)"
    )
    for line in block.splitlines():
        match = label.match(line)
        if match:
            options.update(re.findall(r"--([a-z][a-z0-9-]*)", match.group(1)))
    options.discard("help")  # PowerShell advanced scripts provide common help.
    return options


def _powershell_parameters(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8-sig")
    return {
        _normalise_option(name)
        for name in re.findall(r"\$(\w+)\s*(?:=|,|\))", text)
    }


def parity_errors(root: Path = PROJECT_ROOT) -> list[str]:
    linux_dir = root / "scripts" / "linux"
    windows_dir = root / "scripts" / "windows"
    errors: list[str] = []
    windows_by_stem: dict[str, list[Path]] = {}
    for path in windows_dir.iterdir():
        if path.is_file() and path.suffix.lower() in {".ps1", ".cmd"}:
            windows_by_stem.setdefault(path.stem, []).append(path)

    for linux_path in sorted(linux_dir.glob("*.sh")):
        candidates = windows_by_stem.get(linux_path.stem, [])
        if not candidates:
            errors.append(f"Missing Windows counterpart for {linux_path.name}")
            continue
        powershell = next((path for path in candidates if path.suffix == ".ps1"), None)
        if powershell is None:
            continue  # CMD wrappers forward their argv to a shared implementation.
        windows_parameters = _powershell_parameters(powershell)
        for option in sorted(_linux_public_options(linux_path)):
            if _normalise_option(option) not in windows_parameters:
                errors.append(
                    f"{powershell.name} lacks -{option} from {linux_path.name}"
                )
    return errors


def main() -> int:
    errors = parity_errors()
    if errors:
        print("Linux-to-Windows launcher parity failed:")
        for error in errors:
            print(f"  - {error}")
        return 1
    count = len(tuple(LINUX_DIR.glob("*.sh")))
    print(f"[OK] {count} Linux launchers have Windows counterparts and option parity.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
