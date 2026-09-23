"""Compatibility rules that must hold before constructing datasets or models."""

from config.config import ExperimentConfig


def validate_algorithm_dataset(
    algorithm_key: str, cfg: ExperimentConfig
) -> None:
    """Reject algorithm/dataset combinations that are not supported."""
    if algorithm_key == "mf_hutchinson" and cfg.dataset.name != "celeba_latent":
        raise ValueError(
            "mf_hutchinson is a latent-only diagnostic and requires "
            "dataset.name='celeba_latent'. Use "
            "config/mf_hutchinson_celeba_latent.json."
        )
