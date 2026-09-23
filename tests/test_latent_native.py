from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from algorithms.consistency import ConsistencyAlgorithm
from algorithms.flow_matching import FlowMatchingAlgorithm
from algorithms.flow_matching_lognorm import FlowMatchingLognormAlgorithm
from algorithms.mean_flow import MeanFlowAlgorithm
from algorithms.mean_flow_hutchinson import MeanFlowHutchinsonAlgorithm
from algorithms.mean_flow_distill import MeanFlowDistillAlgorithm
from algorithms.reflow import ReflowAlgorithm
from codec.codec_factory import load_codec
from codec.scratch_vae import ScratchKLVAE
from config.config import BackboneConfig, DatasetConfig, ExperimentConfig
from data.celeba_latent import CelebALatentDataset, sha256_file
from models.backbone import SimpleUNet
from scripts.generate_reflow_pairs_latent import assemble_chunks, integrate_fm


class ConstantBackbone(nn.Module):
    def __init__(self, *, sample_clamp: bool):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))
        self.cfg = SimpleNamespace(
            in_channels=3, sample_clamp=sample_clamp, time_embed_dim=8
        )
        self._expected_image_size = 16

    def forward(self, inputs: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        del time
        return torch.full_like(inputs, -10.0) + self.anchor * 0.0


def _freeze_test_stats(codec: ScratchKLVAE) -> None:
    codec._load_frozen_stats(
        [[[[0.5]], [[-0.25]], [[1.0]], [[0.0]]]],
        [[[[2.0]], [[0.5]], [[1.5]], [[3.0]]]],
        torch.device("cpu"),
    )


def test_scratch_codec_shapes_kl_normalization_and_checkpoint(tmp_path: Path) -> None:
    codec = ScratchKLVAE()
    images = torch.randn(1, 3, 64, 64).clamp(-1, 1)
    sample, mean, logvar = codec.encode(images)
    assert sample.shape == mean.shape == logvar.shape == (1, 4, 16, 16)
    assert codec.encode_mean(images).shape == (1, 4, 16, 16)
    assert codec.decode(mean).shape == (1, 3, 64, 64)
    assert torch.isfinite(codec.kl_loss(mean, logvar))

    _freeze_test_stats(codec)
    latent = torch.randn(2, 4, 16, 16)
    assert torch.allclose(codec.denormalise(codec.normalise(latent)), latent, atol=1e-6)
    checkpoint = tmp_path / "scratch.pt"
    codec.save(checkpoint)
    restored = load_codec(str(checkpoint), torch.device("cpu"))
    assert isinstance(restored, ScratchKLVAE)
    assert restored.metadata()["codec_type"] == "scratch_kl_vae"
    assert restored.stats_frozen
    assert torch.equal(restored.latent_mean, codec.latent_mean)


def test_every_json_config_parses_and_latent_run_names_are_isolated() -> None:
    config_dir = Path(__file__).resolve().parents[1] / "config"
    configs = [ExperimentConfig.load(str(path)) for path in sorted(config_dir.glob("*.json"))]
    latent = [cfg for cfg in configs if cfg.dataset.name == "celeba_latent"]
    canonical = {"fm", "fm_lognorm", "mf", "mf_distill", "consistency", "reflow"}
    assert canonical.issubset({cfg.experiment_name for cfg in latent})
    latent_names = {f"{cfg.experiment_name}_{cfg.dataset.name}" for cfg in latent}
    pixel_names = {
        f"{cfg.experiment_name}_{cfg.dataset.name}"
        for cfg in configs
        if cfg.dataset.name != "celeba_latent"
    }
    assert len(latent_names) == len(latent)
    assert latent_names.isdisjoint(pixel_names)
    assert all(
        cfg.dataset.image_size == 16
        and cfg.backbone.in_channels == 3
        and not cfg.backbone.sample_clamp
        for cfg in latent
    )


def test_mf_hutchinson_executes_on_three_channel_latents() -> None:
    model = SimpleUNet(
        BackboneConfig(
            in_channels=3,
            base_channels=8,
            channel_mults=[1],
            num_res_blocks=1,
            time_embed_dim=8,
            sample_clamp=False,
        )
    )
    model._expected_image_size = 16
    algorithm = MeanFlowHutchinsonAlgorithm(
        model,
        {"p_same": 0.1, "p_hutchinson_step": 1.0, "n_probes": 1},
    )
    latent_batch = torch.randn(2, 3, 16, 16)

    loss = algorithm.training_step(latent_batch)["loss"]
    loss.backward()
    samples = algorithm.sample(2, 1, torch.device("cpu"))

    assert torch.isfinite(loss)
    assert samples.shape == latent_batch.shape
    assert torch.isfinite(samples).all()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_mf_hutchinson_rejects_pixel_backbone() -> None:
    model = SimpleUNet(
        BackboneConfig(
            in_channels=3,
            base_channels=8,
            channel_mults=[1],
            num_res_blocks=1,
            time_embed_dim=8,
            sample_clamp=True,
        )
    )
    with pytest.raises(ValueError, match="pixel-space models are not supported"):
        MeanFlowHutchinsonAlgorithm(model)


def _write_cache(
    root: Path,
    checkpoint: Path,
    *,
    split: str = "train",
    tensor: torch.Tensor | None = None,
    codec_digest: str | None = None,
    corrupt_hash: bool = False,
) -> DatasetConfig:
    tensor = tensor if tensor is not None else torch.randn(3, 4, 16, 16)
    cache = root / split
    cache.mkdir(parents=True, exist_ok=True)
    tensor_path = cache / f"latents_{split}.pt"
    torch.save(tensor, tensor_path)
    digest = sha256_file(tensor_path)
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    codec_metadata = checkpoint_payload["metadata"]
    content_digest = hashlib.sha256()
    fields = (
        ("weights_sha256", codec_metadata["codec_weights_sha256"]),
        ("source_revision", codec_metadata["codec_source_revision"]),
        ("split", split), ("posterior_mode", "mean"),
        ("normalization_schema", "v1_mean_std_frozen_train"),
        ("preprocessing_schema", "celeba_center_crop_178_resize_64_norm_m1p1"),
        ("latent_channels", 4), ("spatial_factor", 4), ("pixel_size", 64),
    )
    for label, value in fields:
        content_digest.update(
            label.encode("utf-8") + b"\x00" + str(value).encode("utf-8") + b"\x00"
        )
    manifest = {
        "schema_version": 1,
        "content_hash": content_digest.hexdigest(),
        "splits": [split],
        "codec_weights_sha256": codec_digest or codec_metadata["codec_weights_sha256"],
        "codec_source": codec_metadata["codec_source"],
        "codec_source_revision": codec_metadata["codec_source_revision"],
        "file_hashes": {tensor_path.name: "0" * 64 if corrupt_hash else digest},
        "posterior_mode": "mean",
        "latent_channels": 4,
        "spatial_factor": 4,
        "pixel_size": 64,
        "normalization_schema": "v1_mean_std_frozen_train",
        "preprocessing_schema": "celeba_center_crop_178_resize_64_norm_m1p1",
    }
    (cache / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return DatasetConfig(
        name="celeba_latent", image_size=16, cache_dir=str(root),
        codec_checkpoint=str(checkpoint), split=split, num_workers=0,
    )


def test_latent_dataset_validates_cache_and_returns_label(tmp_path: Path) -> None:
    checkpoint = tmp_path / "codec.pt"
    codec = ScratchKLVAE()
    _freeze_test_stats(codec)
    codec.save(checkpoint)
    cfg = _write_cache(tmp_path / "cache", checkpoint)
    dataset = CelebALatentDataset(cfg)
    latent, label = dataset[0]
    assert latent.shape == (4, 16, 16)
    assert label == -1


@pytest.mark.parametrize("fault", ["shape", "hash", "identity"])
def test_latent_dataset_rejects_invalid_cache(tmp_path: Path, fault: str) -> None:
    checkpoint = tmp_path / "codec.pt"
    codec = ScratchKLVAE()
    _freeze_test_stats(codec)
    codec.save(checkpoint)
    tensor = torch.randn(2, 3 if fault == "shape" else 4, 16, 16)
    cfg = _write_cache(
        tmp_path / "cache", checkpoint, tensor=tensor,
        corrupt_hash=fault == "hash", codec_digest="f" * 64 if fault == "identity" else None,
    )
    with pytest.raises((ValueError, FileNotFoundError)):
        CelebALatentDataset(cfg)


def _algorithm_pair(cls, tmp_path: Path):
    kwargs = {}
    if cls in (MeanFlowDistillAlgorithm, ConsistencyAlgorithm):
        teacher = tmp_path / "teacher.pt"
        teacher.touch()
        kwargs["teacher_checkpoint"] = str(teacher)
    if cls is ReflowAlgorithm:
        pairs = tmp_path / "pairs.pt"
        torch.save(
            {"z1": torch.randn(2, 3, 16, 16), "x0": torch.randn(2, 3, 16, 16)},
            pairs,
        )
        kwargs["pairs_path"] = str(pairs)
    algorithms = (
        cls(ConstantBackbone(sample_clamp=True), kwargs),
        cls(ConstantBackbone(sample_clamp=False), kwargs),
    )
    if cls is MeanFlowAlgorithm:
        for algorithm in algorithms:
            algorithm._forward = (
                lambda inputs, r, time, model=algorithm.model: model(inputs, time)
            )
    if cls is MeanFlowDistillAlgorithm:
        for algorithm in algorithms:
            algorithm._student_forward = (
                lambda inputs, r, time, model=algorithm.model: model(inputs, time)
            )
    return algorithms


@pytest.mark.parametrize(
    "algorithm_cls",
    [FlowMatchingAlgorithm, FlowMatchingLognormAlgorithm, MeanFlowAlgorithm,
     MeanFlowDistillAlgorithm, ConsistencyAlgorithm, ReflowAlgorithm],
)
def test_all_algorithms_preserve_pixel_clamp_and_leave_latents_unbounded(
    tmp_path: Path, algorithm_cls
) -> None:
    pixel, latent = _algorithm_pair(algorithm_cls, tmp_path)
    torch.manual_seed(123)
    pixel_sample = pixel.sample(2, 1, torch.device("cpu"))
    torch.manual_seed(123)
    latent_sample = latent.sample(2, 1, torch.device("cpu"))
    assert pixel_sample.abs().max() <= 1
    assert torch.isfinite(latent_sample).all()
    assert latent_sample.abs().max() > 1


def test_reflow_mock_integration_and_chunk_assembly(tmp_path: Path) -> None:
    model = ConstantBackbone(sample_clamp=False)
    z1 = torch.zeros(2, 3, 16, 16)
    x0 = integrate_fm(model, z1, nfe=2)
    assert torch.equal(x0, torch.full_like(x0, 10.0))

    chunk_dir = tmp_path / "chunks"
    chunk_dir.mkdir()
    for start, end in ((0, 2), (2, 3)):
        count = end - start
        torch.save(
            {"start": start, "end": end,
             "z1": torch.full((count, 3, 16, 16), float(start)),
             "x0": torch.full((count, 3, 16, 16), float(end))},
            chunk_dir / f"chunk_{start:08d}_{end:08d}.pt",
        )
    payload = assemble_chunks(chunk_dir, {"pair_count": 3, "chunk_size": 2})
    assert set(payload) == {"z1", "x0", "metadata"}
    assert payload["z1"].shape == payload["x0"].shape == (3, 3, 16, 16)
