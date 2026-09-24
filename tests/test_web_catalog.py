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
    assert model["backbone_label"].startswith("unknown")
    assert model["training_history"][0]["epoch"] == 40
    assert model["evaluations"] == [
        {
            "epoch": 40,
            "nfe": 20,
            "fid": 12.5,
            "is_mean": None,
            "is_std": None,
            "num_generated_samples": None,
            "backbone_seconds": None,
            "decoder_seconds": None,
        }
    ]
    assert model["sampling_history"] == []
    assert model["cached_samples"] == []
    assert model["evidence"]["seed"] == 0
    assert model["evidence"]["protocol_key"]


def test_catalog_discovers_checkpoint_derived_sample_artifact(tmp_path: Path) -> None:
    run = tmp_path / "fm_celeba"
    checkpoint_dir = run / "checkpoints" / "run_1"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "FlowMatchingAlgorithm_epoch40.pt").touch()
    (run / "config.json").write_text(
        json.dumps(
            {
                "experiment_name": "fm",
                "dataset": {"name": "celeba", "image_size": 64},
                "evaluation": {"nfe_values": [1, 20]},
            }
        ),
        encoding="utf-8",
    )
    artifact_dir = tmp_path / "checkpoint_samples" / "fm_celeba"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "epoch040_nfe20_seed7.png").write_bytes(b"png")
    (artifact_dir / "epoch040_nfe20_seed7.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "checkpoint_sample_grid",
                "experiment": "fm_celeba",
                "checkpoint": "FlowMatchingAlgorithm_epoch40.pt",
                "dataset": "celeba",
                "epoch": 40,
                "nfe": 20,
                "seed": 7,
                "num_images": 64,
            }
        ),
        encoding="utf-8",
    )

    server.MODEL_SPECS = server.build_model_specs(tmp_path)
    model = server.model_catalog()[0]

    assert model["cached_samples"] == [
        {
            "epoch": 40,
            "nfe": 20,
            "seed": 7,
            "num_images": 64,
            "checkpoint": "FlowMatchingAlgorithm_epoch40.pt",
            "dataset": "celeba",
            "source": "checkpoint_cache",
            "url": "/api/cached-sample?model=fm_celeba&epoch=40&nfe=20&seed=7",
        }
    ]


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


def test_inference_ui_exposes_training_and_cached_checkpoint_playback() -> None:
    html = (server.WEB_ROOT / "inference_ui.html").read_text(encoding="utf-8")
    assert "Training: image → noise" in html
    assert "animateEpochs" in html
    assert "Inference: noise → image" in html
    assert "replayCachedArtifact" in html
    assert "without using the GPU" in html
    assert 'fetch("/api/generate"' not in html
    assert f"UI_SCHEMA_VERSION={server.UI_SCHEMA_VERSION}" in html
    assert "Hard-refresh this page" in html
    playback = html.split("async function animateEpochs", 1)[1].split(
        "async function copyEvidence", 1
    )[0]
    assert "fetch(" not in playback
    assert "generate(" not in playback
    assert "Noise → generated images" in html
    assert "loss, FID, and IS advance" in html
    assert '<label for="seed">Generation seed</label><select id="seed">' in html
    assert 'id="seed" type="number"' not in html
    assert "include_trajectory" not in html
    assert "FID vs NFE" in html
    assert "Backbone" in html
    assert 'for="run">Run' in html
    assert "lossRevealEpoch" in html
    assert "Thesis evidence" in html
    assert "Strictly compatible runs" in html
    assert "mixed_code_identity" in html
    assert server.RUNTIME is None
