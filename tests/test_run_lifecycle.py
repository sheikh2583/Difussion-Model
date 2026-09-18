from __future__ import annotations

import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path

from utils.run_lifecycle import (
    archive_existing_run,
    has_run_artifacts,
    latest_checkpoint,
    run_directory,
)


@dataclass
class _Dataset:
    name: str = "cifar10"


@dataclass
class _Config:
    experiment_name: str = "fm"
    output_dir: str = "results"
    dataset: _Dataset = field(default_factory=_Dataset)


class RunLifecycleTests(unittest.TestCase):
    def test_run_directory_is_derived_from_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertEqual(
                run_directory(_Config(), root),
                (root / "results" / "fm_cifar10").resolve(),
            )

    def test_latest_checkpoint_uses_numeric_epoch_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            checkpoint_dir = run_dir / "checkpoints"
            checkpoint_dir.mkdir()
            for epoch in (10, 100, 90):
                (checkpoint_dir / f"Model_epoch{epoch}.pt").touch()
            latest = latest_checkpoint(run_dir)
            self.assertIsNotNone(latest)
            self.assertEqual(latest[0], 100)

    def test_fresh_archives_instead_of_deleting(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "results" / "fm_cifar10"
            run_dir.mkdir(parents=True)
            (run_dir / "config.json").write_text("{}", encoding="utf-8")

            archived = archive_existing_run(run_dir, timestamp="20260918_120000")

            self.assertFalse(run_dir.exists())
            self.assertIsNotNone(archived)
            self.assertTrue((archived / "config.json").is_file())
            self.assertFalse(has_run_artifacts(run_dir))

    def test_path_components_cannot_escape_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = _Config(experiment_name="../escape")
            with self.assertRaises(ValueError):
                run_directory(config, Path(temporary))


if __name__ == "__main__":
    unittest.main()
