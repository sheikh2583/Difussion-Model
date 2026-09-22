import json
from pathlib import Path

import torch

import web.inference_server as server


def test_catalog_uses_config_names_and_numbered_checkpoints(tmp_path: Path) -> None:
    run = tmp_path / "custom_directory_name"
    checkpoint_dir = run / "checkpoints" / "run_2"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "FlowMatchingLognormAlgorithm_epoch20.pt").touch()
    (checkpoint_dir / "FlowMatchingLognormAlgorithm_epoch40.pt").touch()
    (run / "config.json").write_text(
        json.dumps(
            {
                "experiment_name": "fm_lognorm",
                "dataset": {"name": "celeba", "image_size": 64},
                "evaluation": {"nfe_values": [1, 5, 20, 50]},
            }
        ),
        encoding="utf-8",
    )
    metrics = run / "metrics"
    metrics.mkdir()
    (metrics / "custom_directory_name.jsonl").write_text(
        "\n".join(
            (
                json.dumps({"record_type": "train_epoch", "epoch": 40, "loss": 0.125}),
                json.dumps({"record_type": "evaluation", "epoch": 40, "nfe": 20, "fid": 12.5}),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    server.MODEL_SPECS = server.build_model_specs(tmp_path)
    catalog = server.model_catalog()

    assert len(catalog) == 1
    model = catalog[0]
    assert model["key"] == "custom_directory_name"
    assert model["label"] == "Flow Matching + Logit-Normal · CelebA"
    assert model["short_label"] == "FM-LN"
    assert model["algorithm"] == "fm_lognorm"
    assert model["dataset"] == "celeba"
    assert model["representation"] == "pixel"
    assert model["epochs"] == [20, 40]
    assert model["nfe_values"] == [1, 5, 20, 50]
    assert model["latest_loss"] == 0.125
    assert model["best_fid"] == 12.5


class _MockCodec:
    def __init__(self) -> None:
        self.batch_sizes = []

    def decode_normalised(self, latents: torch.Tensor) -> torch.Tensor:
        self.batch_sizes.append(latents.shape[0])
        return torch.zeros(latents.shape[0], 3, 64, 64)


def test_latent_inference_output_is_decoded_in_bounded_batches() -> None:
    codec = _MockCodec()
    latents = torch.randn(5, 3, 16, 16)
    images = server.decode_samples_for_display(latents, codec, batch_size=2)

    assert images.shape == (5, 3, 64, 64)
    assert codec.batch_sizes == [2, 2, 1]


def test_inference_ui_restores_checkpoint_playback() -> None:
    html = (server.WEB_ROOT / "inference_ui.html").read_text(encoding="utf-8")
    assert "Play checkpoint evolution" in html
    assert "playCheckpointEvolution" in html
    assert "model.epochs" in html
    assert "selected NFE" in html
