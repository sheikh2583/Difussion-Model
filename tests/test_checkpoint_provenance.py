from __future__ import annotations

import copy
import tempfile
import unittest
import warnings
from pathlib import Path

from algorithms.mean_flow import MeanFlowAlgorithm
from config.config import ExperimentConfig
from utils.checkpoint_provenance import (
    ResumeCompatibilityError,
    build_provenance,
    validate_provenance,
)


class CheckpointProvenanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = ExperimentConfig(experiment_name="mf_v2", epochs=100)
        self.cfg.algorithm_kwargs = {"p_same": 0.25, "nested": {"b": 2, "a": 1}}
        self.expected = build_provenance(self.cfg, MeanFlowAlgorithm, "mf")

    def test_is_deterministic_and_ignores_target_epoch(self) -> None:
        extended = copy.deepcopy(self.cfg)
        extended.epochs = 150
        extended.dataset.root = "/portable/other/root"
        extended.dataset.num_workers = 0
        extended.algorithm_kwargs = {"nested": {"a": 1, "b": 2}, "p_same": 0.25}
        self.assertEqual(
            self.expected,
            build_provenance(extended, MeanFlowAlgorithm, "mf"),
        )

    def test_mismatch_is_rejected_with_field_name(self) -> None:
        actual = copy.deepcopy(self.expected)
        actual["identity"]["dataset"]["image_size"] = 64
        with self.assertRaisesRegex(ResumeCompatibilityError, "dataset.image_size"):
            validate_provenance(actual, self.expected, Path("wrong.pt"))

    def test_legacy_checkpoint_warns_and_remains_loadable(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            verified = validate_provenance(None, self.expected, Path("legacy.pt"))
        self.assertFalse(verified)
        self.assertIn("Legacy checkpoint", str(caught[0].message))

    def test_matching_metadata_is_accepted(self) -> None:
        self.assertTrue(
            validate_provenance(self.expected, self.expected, Path("matching.pt"))
        )

    def test_source_mismatch_requires_explicit_override(self) -> None:
        actual = copy.deepcopy(self.expected)
        actual["source_identity_sha256"] = "teacher-source"
        expected = copy.deepcopy(self.expected)
        expected["source_identity_sha256"] = "current-source"

        with self.assertRaisesRegex(ResumeCompatibilityError, "source identity"):
            validate_provenance(actual, expected, Path("teacher.pt"))
        with self.assertWarnsRegex(UserWarning, "Accepted source identity mismatch"):
            self.assertTrue(
                validate_provenance(
                    actual,
                    expected,
                    Path("teacher.pt"),
                    allow_source_identity_mismatch=True,
                )
            )

    def test_source_override_does_not_accept_structural_mismatch(self) -> None:
        actual = copy.deepcopy(self.expected)
        actual["identity"]["dataset"]["image_size"] = 64
        with self.assertRaisesRegex(ResumeCompatibilityError, "dataset.image_size"):
            validate_provenance(
                actual,
                self.expected,
                Path("wrong.pt"),
                allow_source_identity_mismatch=True,
            )


if __name__ == "__main__":
    unittest.main()
