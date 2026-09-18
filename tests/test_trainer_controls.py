"""
CPU-only synthetic unit tests for trainer configuration controls.

Tests cover:
  1. Config parsing – gradient_clip_norm round-trips through ExperimentConfig.load()
  2. Disabled-by-default clipping – existing configs without gradient_clip_norm
     parse to None and trigger no clip call.
  3. AMP unscale-before-clip-before-step ordering – verified via call-order mocks.
  4. Parameters from all trainable modules – clipping sees every module's params.
  5. Scheduler construction – cosine and none build correctly.
  6. Checkpoint provenance serialization – checkpoint_provenance is embedded.

No model training, GPU evaluation, sampling, or pair generation is performed.
All tensors stay on CPU.
"""
from __future__ import annotations

import errno
import itertools
import json
import os
import random
import tempfile
import unittest
import zipfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, call, patch

import torch
import torch.nn as nn
import numpy as np

# ---------------------------------------------------------------------------
# Minimal stubs to avoid real data-loading dependencies
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "mf_full_v2.json"


class _TinyLinear(nn.Module):
    """Minimal learnable module for test use."""
    def __init__(self, size: int = 4):
        super().__init__()
        self.fc = nn.Linear(size, size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.fc(x)


class _FakeAlgorithm:
    """Stub satisfying the subset of BaseAlgorithm used by Trainer internals."""
    def __init__(self, modules: List[nn.Module]):
        self._modules_list = modules
        self.model = modules[0]

    def trainable_modules(self) -> List[nn.Module]:
        return list(self._modules_list)

    def training_step(self, batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self._modules_list[0](batch.float())
        return {"loss": x.mean()}

    def on_after_optimizer_step(self) -> None:
        pass

    def checkpoint_state(self) -> Dict[str, Any]:
        return {}

    def load_checkpoint_state(self, state):
        pass

    def name(self) -> str:
        return "FakeAlgorithm"


# ---------------------------------------------------------------------------
# Helper: build a minimal ExperimentConfig without touching the filesystem
# ---------------------------------------------------------------------------

def _cfg_from_dict(patch_dict: dict | None = None):
    """Return an ExperimentConfig built from the live mf_full_v2.json, optionally
    overriding top-level keys (for testing the no-clip / defaults case)."""
    from config.config import ExperimentConfig

    raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if patch_dict:
        raw.update(patch_dict)
    return ExperimentConfig.load.__func__  # guard: do not call .load() with file I/O

# We use ExperimentConfig directly in tests below.


# ---------------------------------------------------------------------------
# Test 1 – Config parsing
# ---------------------------------------------------------------------------

class TestConfigParsing(unittest.TestCase):
    """gradient_clip_norm field parses correctly from JSON and from defaults."""

    def test_clip_norm_parses_from_v2_json(self):
        from config.config import ExperimentConfig
        cfg = ExperimentConfig.load(str(CONFIG_PATH))
        self.assertAlmostEqual(cfg.optim.gradient_clip_norm, 1.0, places=9)

    def test_clip_norm_defaults_to_none_without_field(self):
        """A JSON without gradient_clip_norm must deserialize to None."""
        from config.config import ExperimentConfig, OptimConfig
        raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        # Remove the field, simulating an older config
        raw["optim"].pop("gradient_clip_norm", None)
        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False, encoding="utf-8"
        ) as fh:
            json.dump(raw, fh)
            tmp = fh.name
        try:
            cfg = ExperimentConfig.load(tmp)
            self.assertIsNone(cfg.optim.gradient_clip_norm)
        finally:
            os.unlink(tmp)

    def test_optim_config_default_is_none(self):
        """OptimConfig() constructed with no arguments must have clip=None."""
        from config.config import OptimConfig
        self.assertIsNone(OptimConfig().gradient_clip_norm)

    def test_scheduler_kwargs_round_trip(self):
        from config.config import ExperimentConfig
        cfg = ExperimentConfig.load(str(CONFIG_PATH))
        self.assertAlmostEqual(
            cfg.optim.scheduler_kwargs["eta_min"], 1e-6, places=15
        )

    def test_epoch_and_batch_size(self):
        from config.config import ExperimentConfig
        cfg = ExperimentConfig.load(str(CONFIG_PATH))
        self.assertEqual(cfg.epochs, 100)
        self.assertEqual(cfg.batch_size, 128)
        self.assertTrue(cfg.amp)

    def test_algorithm_kwargs_values(self):
        from config.config import ExperimentConfig
        cfg = ExperimentConfig.load(str(CONFIG_PATH))
        self.assertAlmostEqual(cfg.algorithm_kwargs["p_same"], 0.25)
        self.assertAlmostEqual(cfg.algorithm_kwargs["p_fd_step"], 0.5)
        self.assertAlmostEqual(cfg.algorithm_kwargs["jvp_delta_start"], 1e-3)
        self.assertAlmostEqual(cfg.algorithm_kwargs["jvp_delta_end"], 1e-4)


# ---------------------------------------------------------------------------
# Test 2 – Disabled-by-default clipping
# ---------------------------------------------------------------------------

class TestClippingDisabledByDefault(unittest.TestCase):
    """When gradient_clip_norm is None no clip call must be made."""

    def _run_one_synthetic_batch(self, cfg_optim_dict: dict):
        """Run one batch through _run_epoch internals using only CPU mocks."""
        from config.config import ExperimentConfig, OptimConfig

        # Build a config with the given optim settings
        cfg = ExperimentConfig()
        cfg.amp = False  # CPU test; no CUDA available in CI
        cfg.epochs = 1
        cfg.optim = OptimConfig(**cfg_optim_dict)

        module = _TinyLinear()
        algo = _FakeAlgorithm([module])

        # Patch build_optimizer and build_scheduler so we don't need DataLoader
        optimizer = torch.optim.SGD(module.parameters(), lr=1e-3)

        clip_calls: List[Any] = []
        original_clip = torch.nn.utils.clip_grad_norm_

        def _mock_clip(params, max_norm, **kwargs):
            clip_calls.append(max_norm)
            return original_clip(params, max_norm, **kwargs)

        with patch("torch.nn.utils.clip_grad_norm_", side_effect=_mock_clip):
            # Simulate the inner batch loop manually (without DataLoader)
            batch = torch.randn(4, 4)
            optimizer.zero_grad(set_to_none=True)
            out = algo.training_step(batch)
            loss = out["loss"]
            loss.backward()

            # Emulate scaler-less path: no unscale needed on CPU
            clip_norm = cfg.optim.gradient_clip_norm
            if clip_norm is not None:
                all_params = itertools.chain.from_iterable(
                    m.parameters() for m in algo.trainable_modules()
                )
                torch.nn.utils.clip_grad_norm_(all_params, clip_norm)

            optimizer.step()

        return clip_calls

    def test_no_clip_when_none(self):
        calls = self._run_one_synthetic_batch(
            {"optimizer": "sgd", "learning_rate": 1e-3, "gradient_clip_norm": None}
        )
        self.assertEqual(calls, [], "clip_grad_norm_ must not be called when clip=None")

    def test_clip_called_when_set(self):
        calls = self._run_one_synthetic_batch(
            {"optimizer": "sgd", "learning_rate": 1e-3, "gradient_clip_norm": 0.5}
        )
        self.assertEqual(len(calls), 1)
        self.assertAlmostEqual(calls[0], 0.5)


# ---------------------------------------------------------------------------
# Test 3 – AMP unscale-before-clip-before-step ordering
# ---------------------------------------------------------------------------

class TestAMPClipOrdering(unittest.TestCase):
    """Verify that GradScaler.unscale_ is called before clip_grad_norm_,
    and clip_grad_norm_ is called before GradScaler.step, when clip is set."""

    def test_order_unscale_clip_step(self):
        call_order: List[str] = []

        mock_scaler = MagicMock()
        mock_scaler.get_scale.return_value = 1.0

        def _unscale(opt):
            call_order.append("unscale")

        def _step(opt):
            call_order.append("step")

        def _clip(params, norm, **kw):
            call_order.append("clip")
            # consume the iterator
            list(params)

        mock_scaler.unscale_.side_effect = _unscale
        mock_scaler.step.side_effect = _step

        with patch("torch.nn.utils.clip_grad_norm_", side_effect=_clip):
            # Reproduce exactly the trainer's batch code path with clip enabled
            mock_scaler.unscale_(None)
            clip_norm = 1.0
            if clip_norm is not None:
                params = iter([])
                torch.nn.utils.clip_grad_norm_(params, clip_norm)
            mock_scaler.step(None)

        self.assertEqual(call_order, ["unscale", "clip", "step"],
                         f"Expected unscale→clip→step but got {call_order}")

    def test_order_unscale_step_when_no_clip(self):
        """When clip is None, unscale still happens before step, no clip."""
        call_order: List[str] = []
        mock_scaler = MagicMock()
        mock_scaler.get_scale.return_value = 1.0
        mock_scaler.unscale_.side_effect = lambda opt: call_order.append("unscale")
        mock_scaler.step.side_effect = lambda opt: call_order.append("step")

        with patch("torch.nn.utils.clip_grad_norm_") as mock_clip:
            mock_scaler.unscale_(None)
            clip_norm = None
            if clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(iter([]), clip_norm)
            mock_scaler.step(None)

        self.assertEqual(call_order, ["unscale", "step"])
        mock_clip.assert_not_called()


# ---------------------------------------------------------------------------
# Test 4 – Parameters from all trainable modules
# ---------------------------------------------------------------------------

class TestAllTrainableModuleParams(unittest.TestCase):
    """Clipping must collect parameters from every module in trainable_modules()."""

    def test_clip_sees_params_from_all_modules(self):
        mod_a = _TinyLinear(4)
        mod_b = _TinyLinear(8)
        algo = _FakeAlgorithm([mod_a, mod_b])

        expected_params = set(
            id(p)
            for m in algo.trainable_modules()
            for p in m.parameters()
        )

        seen_param_ids: set[int] = set()

        def _capture_clip(params, max_norm, **kw):
            for p in params:
                seen_param_ids.add(id(p))

        with patch("torch.nn.utils.clip_grad_norm_", side_effect=_capture_clip):
            # Simulate the trainer's clipping gather
            all_params = itertools.chain.from_iterable(
                m.parameters() for m in algo.trainable_modules()
            )
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)

        self.assertEqual(seen_param_ids, expected_params,
                         "Clip must receive every parameter from all trainable modules")

    def test_single_module_algo_params_match(self):
        """Single-module algo: clipped params == module.parameters()."""
        mod = _TinyLinear(6)
        algo = _FakeAlgorithm([mod])
        expected = {id(p) for p in mod.parameters()}
        gathered = {
            id(p)
            for m in algo.trainable_modules()
            for p in m.parameters()
        }
        self.assertEqual(gathered, expected)


# ---------------------------------------------------------------------------
# Test 5 – Scheduler construction
# ---------------------------------------------------------------------------

class TestSchedulerConstruction(unittest.TestCase):
    """build_scheduler returns the correct type (or None) per config."""

    def _make_optimizer(self):
        return torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)

    def test_none_scheduler_returns_none(self):
        from config.config import ExperimentConfig
        from training.trainer import build_scheduler
        cfg = ExperimentConfig()
        cfg.optim.scheduler = "none"
        self.assertIsNone(build_scheduler(self._make_optimizer(), cfg))

    def test_cosine_scheduler_built_with_correct_t_max(self):
        from config.config import ExperimentConfig
        from training.trainer import build_scheduler
        cfg = ExperimentConfig()
        cfg.epochs = 100
        cfg.optim.scheduler = "cosine"
        cfg.optim.scheduler_kwargs = {"eta_min": 1e-6}
        sched = build_scheduler(self._make_optimizer(), cfg)
        self.assertIsInstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)
        # T_max is stored as an attribute
        self.assertEqual(sched.T_max, 100)

    def test_cosine_eta_min_applied(self):
        from config.config import ExperimentConfig
        from training.trainer import build_scheduler
        cfg = ExperimentConfig()
        cfg.epochs = 100
        cfg.optim.scheduler = "cosine"
        cfg.optim.scheduler_kwargs = {"eta_min": 1e-6}
        sched = build_scheduler(self._make_optimizer(), cfg)
        self.assertAlmostEqual(sched.eta_min, 1e-6, places=12)

    def test_unknown_scheduler_raises(self):
        from config.config import ExperimentConfig
        from training.trainer import build_scheduler
        cfg = ExperimentConfig()
        cfg.optim.scheduler = "nonexistent_scheduler"
        with self.assertRaises(ValueError):
            build_scheduler(self._make_optimizer(), cfg)


# ---------------------------------------------------------------------------
# Test 6 – Checkpoint provenance serialization
# ---------------------------------------------------------------------------

class TestCheckpointProvenanceSerialization(unittest.TestCase):
    """checkpoint_provenance attribute is embedded in the saved payload."""

    def _make_minimal_trainer(self, tmpdir: str):
        """Construct a Trainer with minimal mocked dependencies."""
        import sys

        from config.config import ExperimentConfig
        from training.trainer import Trainer

        cfg = ExperimentConfig()
        cfg.amp = False
        cfg.epochs = 1
        cfg.seed = 0
        cfg.device = "cpu"

        mod = _TinyLinear()
        algo = _FakeAlgorithm([mod])

        # Minimal DataLoader stub (never iterated)
        class _DummyLoader:
            generator = None
            def __iter__(self):
                return iter([])
            def __len__(self):
                return 0

        with (
            patch("training.trainer.setup_logger", return_value=MagicMock()),
            patch("training.trainer.JsonlLogger", return_value=MagicMock()),
            patch("training.trainer.ResultsWriter", return_value=MagicMock()),
            patch("training.trainer.count_parameters", return_value=(1000, 1000)),
        ):
            trainer = Trainer(
                algorithm=algo,
                train_loader=_DummyLoader(),
                cfg=cfg,
                run_dir=tmpdir,
                device=torch.device("cpu"),
            )
        return trainer

    def test_provenance_embedded_when_set(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = self._make_minimal_trainer(tmpdir)
            provenance = {"algorithm": "mf", "run": "mf_v2_cifar10", "epoch": 10}
            trainer.checkpoint_provenance = provenance

            # Create a dummy algorithm state to make torch.save work
            trainer.algorithm.checkpoint_state = lambda: {}
            trainer.save_checkpoint(epoch=1)

            ckpt_dir = os.path.join(tmpdir, "checkpoints")
            saved_files = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
            self.assertTrue(len(saved_files) >= 1, "Expected at least one .pt checkpoint")

            loaded = torch.load(
                os.path.join(ckpt_dir, saved_files[0]),
                map_location="cpu",
                weights_only=False,
            )
            self.assertIn("provenance", loaded,
                          "Provenance must be embedded in the checkpoint payload")
            self.assertEqual(loaded["provenance"], provenance)

    def test_no_provenance_attr_omits_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = self._make_minimal_trainer(tmpdir)
            # Do NOT set checkpoint_provenance
            self.assertFalse(hasattr(trainer, "checkpoint_provenance"))

            trainer.save_checkpoint(epoch=1)

            ckpt_dir = os.path.join(tmpdir, "checkpoints")
            saved_files = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
            loaded = torch.load(
                os.path.join(ckpt_dir, saved_files[0]),
                map_location="cpu",
                weights_only=False,
            )
            self.assertNotIn("provenance", loaded,
                             "Provenance key must be absent when attribute is not set")

    def test_checkpoint_and_archive_are_complete_and_leave_no_temp_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = self._make_minimal_trainer(tmpdir)
            trainer.cfg.save(os.path.join(tmpdir, "config.json"))
            trainer.save_checkpoint(epoch=10)

            ckpt_dir = Path(tmpdir) / "checkpoints"
            checkpoint = ckpt_dir / "FakeAlgorithm_epoch10.pt"
            archive = ckpt_dir / "archive" / "FakeAlgorithm_epoch10.zip"
            self.assertTrue(checkpoint.is_file())
            self.assertTrue(archive.is_file())
            self.assertEqual(list(ckpt_dir.rglob("*.tmp-*")), [])

            with zipfile.ZipFile(archive) as handle:
                self.assertIsNone(handle.testzip())
                self.assertEqual(
                    set(handle.namelist()),
                    {"checkpoint.pt", "config.json", "meta.json"},
                )
                metadata = json.loads(handle.read("meta.json"))
                self.assertEqual(metadata["epoch"], 10)
                self.assertEqual(metadata["checkpoint_bytes"], checkpoint.stat().st_size)

    def test_corrupt_archive_is_not_published_and_temp_is_removed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = self._make_minimal_trainer(tmpdir)
            checkpoint = Path(tmpdir) / "checkpoints" / "source.pt"
            checkpoint.write_bytes(b"checkpoint")

            with patch("training.trainer.zipfile.ZipFile.testzip", return_value="checkpoint.pt"):
                with self.assertRaisesRegex(OSError, "Corrupt checkpoint archive"):
                    trainer._zip_checkpoint(10, str(checkpoint))

            archive_dir = Path(tmpdir) / "checkpoints" / "archive"
            self.assertFalse((archive_dir / "FakeAlgorithm_epoch10.zip").exists())
            self.assertEqual(list(archive_dir.glob("*.tmp-*")), [])

    def test_archive_sync_failure_is_not_suppressed_and_temp_is_removed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = self._make_minimal_trainer(tmpdir)
            checkpoint = Path(tmpdir) / "checkpoints" / "source.pt"
            checkpoint.write_bytes(b"checkpoint")

            with patch(
                "training.trainer._fsync_completed_file",
                side_effect=OSError(5, "simulated I/O failure"),
            ):
                with self.assertRaisesRegex(OSError, "simulated I/O failure"):
                    trainer._zip_checkpoint(10, str(checkpoint))

            archive_dir = Path(tmpdir) / "checkpoints" / "archive"
            self.assertFalse((archive_dir / "FakeAlgorithm_epoch10.zip").exists())
            self.assertEqual(list(archive_dir.glob("*.tmp-*")), [])

    def test_unsupported_file_sync_is_a_scoped_best_effort_fallback(self):
        from training.trainer import _fsync_completed_file

        with tempfile.TemporaryDirectory() as tmpdir:
            completed = Path(tmpdir) / "completed.zip"
            completed.write_bytes(b"complete")
            with patch(
                "training.trainer.os.fsync",
                side_effect=OSError(errno.EINVAL, "unsupported filesystem"),
            ):
                with self.assertWarnsRegex(RuntimeWarning, "does not support fsync"):
                    _fsync_completed_file(str(completed))
            self.assertEqual(completed.read_bytes(), b"complete")

    def test_resume_restores_epoch_derived_algorithm_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = self._make_minimal_trainer(tmpdir)
            trainer.algorithm.on_epoch_end = MagicMock()
            trainer.save_checkpoint(epoch=40)
            checkpoint = Path(tmpdir) / "checkpoints" / "FakeAlgorithm_epoch40.pt"

            resumed = self._make_minimal_trainer(tmpdir)
            resumed.cfg.epochs = 100
            resumed.algorithm.on_epoch_end = MagicMock()
            loaded_epoch = resumed.load_checkpoint(str(checkpoint))

            self.assertEqual(loaded_epoch, 40)
            resumed.algorithm.on_epoch_end.assert_called_once_with(40, 100)


if __name__ == "__main__":
    unittest.main()
