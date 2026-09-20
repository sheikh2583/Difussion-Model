from pathlib import Path

from scripts.verify_project_layout import (
    config_errors,
    documentation_errors,
    entrypoint_errors,
    platform_placement_errors,
    report_errors,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_current_repository_layout_is_consistent() -> None:
    assert platform_placement_errors(PROJECT_ROOT) == []
    assert config_errors(PROJECT_ROOT) == []
    assert documentation_errors(PROJECT_ROOT) == []
    assert entrypoint_errors(PROJECT_ROOT) == []
    assert report_errors(PROJECT_ROOT) == []


def test_platform_placement_rejects_scattered_launchers(tmp_path: Path) -> None:
    (tmp_path / "orphan.sh").write_text("#!/usr/bin/env sh\n", encoding="utf-8")
    (tmp_path / "orphan.ps1").write_text("Write-Host test\r\n", encoding="utf-8")

    errors = platform_placement_errors(tmp_path)

    assert any("orphan.sh" in error for error in errors)
    assert any("orphan.ps1" in error for error in errors)
