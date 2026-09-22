from types import SimpleNamespace

import torch

from scripts.generate_checkpoint_samples import decode_for_display, load_latent_decoder


class _MockCodec:
    def __init__(self) -> None:
        self.batch_sizes = []

    def decode_normalised(self, latents: torch.Tensor) -> torch.Tensor:
        self.batch_sizes.append(latents.shape[0])
        return torch.zeros(latents.shape[0], 3, 64, 64)


def test_pixel_samples_are_preserved_as_rgb() -> None:
    samples = torch.randn(2, 3, 32, 32)
    decoded = decode_for_display(samples, codec=None)
    assert torch.equal(decoded, samples)


def test_latents_are_decoded_in_bounded_batches() -> None:
    codec = _MockCodec()
    samples = torch.randn(5, 3, 16, 16)
    decoded = decode_for_display(samples, codec, batch_size=2)
    assert decoded.shape == (5, 3, 64, 64)
    assert codec.batch_sizes == [2, 2, 1]


def test_non_rgb_output_is_rejected() -> None:
    samples = torch.randn(2, 4, 16, 16)
    try:
        decode_for_display(samples, codec=None)
    except ValueError as error:
        assert "RGB tensor" in str(error)
    else:
        raise AssertionError("Expected non-RGB model states to be rejected")


def test_pixel_config_does_not_load_a_codec() -> None:
    cfg = SimpleNamespace(dataset=SimpleNamespace(name="cifar10"))
    assert load_latent_decoder(cfg, torch.device("cpu")) is None
