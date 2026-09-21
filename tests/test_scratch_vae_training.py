from pathlib import Path

import pytest
import torch

from codec.scratch_vae import ScratchKLVAE
from codec.train_scratch_vae import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_GRAD_CLIP_NORM,
    DEFAULT_KL_END,
    DEFAULT_KL_START,
    DEFAULT_KL_WARMUP_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_VALIDATION_SAMPLES,
    DEFAULT_VALIDATE_EVERY,
    DEFAULT_WEIGHT_DECAY,
    _kl_weight,
    _latest_checkpoint,
    parse_args,
)


def test_recommended_training_defaults() -> None:
    args = parse_args([])
    assert args.batch_size == DEFAULT_BATCH_SIZE == 128
    assert args.epochs == DEFAULT_EPOCHS == 60
    assert args.learning_rate == DEFAULT_LEARNING_RATE == 1e-4
    assert args.weight_decay == DEFAULT_WEIGHT_DECAY == 1e-4
    assert args.kl_start == DEFAULT_KL_START == 1e-5
    assert args.kl_end == DEFAULT_KL_END == 1e-4
    assert args.kl_warmup_epochs == DEFAULT_KL_WARMUP_EPOCHS == 20
    assert args.gradient_clip_norm == DEFAULT_GRAD_CLIP_NORM == 1.0
    assert args.validate_every == DEFAULT_VALIDATE_EVERY == 5
    assert args.validation_samples == DEFAULT_VALIDATION_SAMPLES == 5000
    assert args.amp is True


def test_kl_schedule_reaches_end_and_stays_there() -> None:
    assert _kl_weight(0) == pytest.approx(1e-5)
    assert _kl_weight(10) == pytest.approx(5.5e-5)
    assert _kl_weight(20) == pytest.approx(1e-4)
    assert _kl_weight(59) == pytest.approx(1e-4)


def test_amp_kl_math_is_finite_fp32() -> None:
    mean = torch.ones(1, 4, 2, 2, dtype=torch.float16)
    logvar = torch.full_like(mean, 20.0)
    loss = ScratchKLVAE.kl_loss(mean, logvar)
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)


def test_latest_checkpoint_uses_numeric_epoch(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    for epoch in (5, 60, 10):
        (checkpoint_dir / f"scratch_vae_epoch{epoch}.pt").touch()
    (checkpoint_dir / "scratch_vae_epochbad.pt").touch()
    assert _latest_checkpoint(tmp_path) == checkpoint_dir / "scratch_vae_epoch60.pt"


@pytest.mark.parametrize(
    "arguments",
    [
        ["--batch-size", "0"],
        ["--epochs", "0"],
        ["--learning-rate", "0"],
        ["--weight-decay", "-1"],
        ["--gradient-clip-norm", "-1"],
    ],
)
def test_invalid_training_hyperparameters_are_rejected(arguments: list[str]) -> None:
    with pytest.raises(SystemExit):
        parse_args(arguments)
