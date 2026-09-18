from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from utils.checkpoint_runs import (
    checkpoint_run_number_from_path,
    find_checkpoint,
    latest_epoch_checkpoint,
    legacy_migration_plan,
    migrate_legacy_checkpoint_layout,
    next_checkpoint_run_number,
    run_directory_from_checkpoint,
)


class CheckpointRunTests(unittest.TestCase):
    def test_legacy_files_and_archives_move_to_run_1(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "results" / "mf_cifar10"
            root = run_dir / "checkpoints"
            (root / "archive").mkdir(parents=True)
            checkpoint = root / "MeanFlowAlgorithm_epoch100.pt"
            archive = root / "archive" / "MeanFlowAlgorithm_epoch100.zip"
            checkpoint.write_bytes(b"checkpoint")
            archive.write_bytes(b"archive")

            plan = legacy_migration_plan(run_dir)
            self.assertEqual(len(plan), 2)
            migrate_legacy_checkpoint_layout(run_dir)

            moved = root / "run_1" / checkpoint.name
            self.assertEqual(moved.read_bytes(), b"checkpoint")
            self.assertEqual(
                (root / "run_1" / "archive" / archive.name).read_bytes(),
                b"archive",
            )
            self.assertFalse(checkpoint.exists())
            self.assertEqual(run_directory_from_checkpoint(moved), run_dir.resolve())

    def test_migration_refuses_collision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            root = run_dir / "checkpoints"
            root.mkdir(parents=True)
            source = root / "Algo_epoch1.pt"
            source.write_bytes(b"old")
            destination = root / "run_1" / source.name
            destination.parent.mkdir()
            destination.write_bytes(b"keep")
            with self.assertRaises(FileExistsError):
                migrate_legacy_checkpoint_layout(run_dir)
            self.assertEqual(source.read_bytes(), b"old")
            self.assertEqual(destination.read_bytes(), b"keep")

    def test_latest_run_is_selected_without_mixing_epochs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            for number, epochs in ((1, (10, 100)), (2, (5, 20))):
                directory = run_dir / "checkpoints" / f"run_{number}"
                directory.mkdir(parents=True)
                for epoch in epochs:
                    (directory / f"Algo_epoch{epoch}.pt").touch()
            self.assertEqual(latest_epoch_checkpoint(run_dir)[0], 20)
            self.assertEqual(find_checkpoint(run_dir, "Algo", 20).parent.name, "run_2")

    def test_next_number_scans_archived_whole_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "results" / "mf_cifar10"
            archived = run_dir.parent / "history" / "mf_cifar10_20260918_120000"
            (archived / "checkpoints" / "run_3").mkdir(parents=True)
            (archived / "checkpoints" / "run_3" / "Algo_epoch100.pt").touch()
            self.assertEqual(next_checkpoint_run_number(run_dir), 4)

    def test_run_number_is_inferred_from_explicit_checkpoint(self) -> None:
        path = Path("results/fm/checkpoints/run_7/Algo_epoch10.pt")
        self.assertEqual(checkpoint_run_number_from_path(path), 7)


if __name__ == "__main__":
    unittest.main()
