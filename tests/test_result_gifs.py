import json

from scripts.generate_result_gifs import dataset_experiments


def _write_config(root, experiment: str, dataset: str) -> None:
    run = root / experiment
    run.mkdir()
    (run / "config.json").write_text(
        json.dumps({"dataset": {"name": dataset}}), encoding="utf-8"
    )


def test_pixel_and_latent_celeba_animations_remain_separate(tmp_path) -> None:
    _write_config(tmp_path, "fm_celeba", "celeba")
    _write_config(tmp_path, "fm_celeba_latent", "celeba_latent")
    records = [
        {"experiment": "fm_celeba"},
        {"experiment": "fm_celeba_latent"},
    ]

    assert dataset_experiments(tmp_path, records, "celeba") == ("fm_celeba",)
    assert dataset_experiments(tmp_path, records, "celeba_latent") == (
        "fm_celeba_latent",
    )
