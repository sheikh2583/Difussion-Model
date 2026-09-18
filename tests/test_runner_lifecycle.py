from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from experiments.runner import ExperimentRunner


class RunnerLifecycleTests(unittest.TestCase):
    def test_auto_resume_without_matching_algorithm_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint_dir = Path(temporary) / "checkpoints" / "run_1"
            checkpoint_dir.mkdir(parents=True)
            (checkpoint_dir / "OtherAlgorithm_epoch100.pt").touch()
            runner = ExperimentRunner.__new__(ExperimentRunner)
            runner.run_dir = temporary
            runner.checkpoint_run_number = 1
            runner.cfg = SimpleNamespace(epochs=100)
            runner.algorithm = SimpleNamespace(name=lambda: "MeanFlowAlgorithm")
            with self.assertRaisesRegex(FileNotFoundError, "no checkpoint matching"):
                runner._resolve_resume_checkpoint("auto")

    @patch("experiments.runner.Trainer")
    @patch("experiments.runner.ensure_fid_reference")
    def test_train_only_bypasses_all_evaluation_setup(
        self, ensure_reference: Mock, trainer_class: Mock
    ) -> None:
        trainer = trainer_class.return_value
        runner = ExperimentRunner.__new__(ExperimentRunner)
        runner.cfg = SimpleNamespace(
            epochs=1,
            evaluation=SimpleNamespace(eval_frequency_epochs=1),
            save=Mock(),
        )
        runner.test_loader = object()
        runner.train_loader = object()
        runner.device = object()
        runner.algorithm = object()
        runner.run_dir = "unused"
        runner.checkpoint_run_number = 1
        runner.checkpoint_provenance = {"version": 1}
        runner._eval_hook = Mock()

        runner.run_train_only()

        ensure_reference.assert_not_called()
        self.assertIsNone(trainer_class.call_args.kwargs["eval_hook"])
        trainer.fit.assert_called_once_with(start_epoch=1)


if __name__ == "__main__":
    unittest.main()
