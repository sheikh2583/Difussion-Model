"""CPU-only tests for explicit pretrained-codec quality acceptance policy."""

import pytest

from codec.validate_codec import _parse_args, _resolve_acceptance


def test_quality_pass_needs_no_override() -> None:
    assert _resolve_acceptance(
        [], accept_quality_failure=False, acceptance_reason=""
    ) == (True, "quality_gate_passed")


def test_quality_failure_is_rejected_by_default() -> None:
    assert _resolve_acceptance(
        ["rFID failed"], accept_quality_failure=False, acceptance_reason=""
    ) == (False, "quality_gate_rejected")


def test_quality_failure_can_be_explicitly_accepted() -> None:
    assert _resolve_acceptance(
        ["rFID failed"],
        accept_quality_failure=True,
        acceptance_reason="Proceed with the selected latent-space experiment",
    ) == (True, "explicit_operator_override")


def test_override_requires_recorded_reason() -> None:
    with pytest.raises(ValueError, match="acceptance-reason"):
        _resolve_acceptance(
            ["PSNR failed"], accept_quality_failure=True, acceptance_reason="  "
        )


def test_cli_rejects_override_without_reason() -> None:
    with pytest.raises(SystemExit):
        _parse_args(["--codec-source-path", "unused", "--accept-quality-failure"])


def test_cli_accepts_audited_override() -> None:
    args = _parse_args(
        [
            "--codec-source-path",
            "unused",
            "--accept-quality-failure",
            "--acceptance-reason",
            "selected latent experiment",
        ]
    )
    assert args.accept_quality_failure
    assert args.acceptance_reason == "selected latent experiment"
