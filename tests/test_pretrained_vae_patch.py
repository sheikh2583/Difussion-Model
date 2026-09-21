"""CPU-only regression tests for pretrained-codec source-path persistence."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import codec.pretrained_vae as pretrained_module
from codec.pretrained_vae import PretrainedVQCodec


class _MockVQModel(nn.Module):
    """Tiny shape-compatible stand-in; no weights or network access required."""

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.25))

    @classmethod
    def from_pretrained(cls, source_path: str):
        del source_path
        return cls()

    def encode(self, images: torch.Tensor):
        latents = torch.zeros(
            images.shape[0], 3, 16, 16, device=images.device, dtype=images.dtype
        ) + self.anchor * 0.0
        return SimpleNamespace(latents=latents)

    def quantize(self, latents: torch.Tensor):
        return latents, torch.zeros((), device=latents.device), None

    def decode(self, latents: torch.Tensor, force_not_quantize: bool = False):
        del force_not_quantize
        sample = torch.zeros(
            latents.shape[0], 3, 64, 64, device=latents.device, dtype=latents.dtype
        ) + self.anchor * 0.0
        return SimpleNamespace(sample=sample)


@pytest.fixture
def mocked_diffusers(monkeypatch):
    monkeypatch.setattr(pretrained_module, "_DIFFUSERS_AVAILABLE", True)
    monkeypatch.setattr(pretrained_module, "_VQModel", _MockVQModel)


def _frozen_codec(source_dir) -> PretrainedVQCodec:
    codec = PretrainedVQCodec.from_pretrained(
        source_path=str(source_dir),
        device=torch.device("cpu"),
        codec_source_revision="mock-revision",
    )
    codec._load_frozen_stats(
        [[[[0.0]], [[0.0]], [[0.0]]]],
        [[[[1.0]], [[1.0]], [[1.0]]]],
        torch.device("cpu"),
    )
    return codec


def test_source_path_survives_save_and_reload(tmp_path, mocked_diffusers) -> None:
    source_dir = tmp_path / "mock-vae"
    source_dir.mkdir()
    checkpoint = tmp_path / "accepted_codec.pt"

    codec = _frozen_codec(source_dir)
    assert codec._source_path == str(source_dir.resolve())
    codec.save(str(checkpoint))

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["source_path"] == str(source_dir.resolve())
    restored = PretrainedVQCodec.from_checkpoint(checkpoint, torch.device("cpu"))
    assert restored._source_path == str(source_dir.resolve())
    assert restored.stats_frozen


def test_validation_acceptance_is_embedded_in_checkpoint(tmp_path, mocked_diffusers) -> None:
    source_dir = tmp_path / "mock-vae"
    source_dir.mkdir()
    checkpoint = tmp_path / "accepted_codec.pt"
    validation = {
        "quality_gate_passed": False,
        "accepted_for_latent_training": True,
        "acceptance_policy": "explicit_operator_override",
        "acceptance_reason": "selected latent experiment",
        "failures": ["rFID failed", "PSNR failed"],
    }

    _frozen_codec(source_dir).save(
        str(checkpoint), validation_metadata=validation
    )

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert payload["validation"] == validation


def test_save_rejects_missing_source_path(tmp_path, mocked_diffusers) -> None:
    source_dir = tmp_path / "mock-vae"
    source_dir.mkdir()
    codec = _frozen_codec(source_dir)
    del codec._source_path

    with pytest.raises(RuntimeError, match="source_path"):
        codec.save(str(tmp_path / "invalid.pt"))
