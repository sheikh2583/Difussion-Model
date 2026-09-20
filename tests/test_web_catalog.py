import json
from pathlib import Path

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
    assert model["epochs"] == [20, 40]
    assert model["nfe_values"] == [1, 5, 20, 50]
    assert model["latest_loss"] == 0.125
    assert model["best_fid"] == 12.5
