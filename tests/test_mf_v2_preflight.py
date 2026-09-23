from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.preflight_mf_v2 import PreflightError, validate_mf_v2, verify_mf_v3


def _valid_config() -> dict:
    return {
        "experiment_name": "mf_v2",
        "output_dir": "./results",
        "amp": True,
        "batch_size": 128,
        "epochs": 100,
        "dataset": {"name": "cifar10", "image_size": 32},
        "optim": {
            "optimizer": "adamw",
            "learning_rate": 1e-4,
            "weight_decay": 1e-4,
            "gradient_clip_norm": 1.0,
            "scheduler": "cosine",
            "scheduler_kwargs": {"eta_min": 1e-6},
        },
        "evaluation": {
            "num_generated_samples": 5000,
            "nfe_values": [1, 2, 5, 10, 20],
            "fid_reference_cache": "./results/metrics/fid_reference_stats_5000_v2.npz",
        },
        "algorithm_kwargs": {
            "p_same": 0.25,
            "p_fd_step": 0.5,
            "jvp_delta_start": 1e-3,
            "jvp_delta_end": 1e-4,
        },
    }


class MfV2PreflightTests(unittest.TestCase):
    def _root(self, temporary: str) -> Path:
        root = Path(temporary)
        (root / "config").mkdir()
        (root / "config" / "mf_full.json").write_text(
            json.dumps({"evaluation": {"fid_reference_cache": "old-cache.npz"}}),
            encoding="utf-8",
        )
        return root

    def test_fresh_clone_passes_with_evidence_warning(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            warnings = validate_mf_v2(_valid_config(), self._root(temporary))
        self.assertEqual(len(warnings), 1)
        self.assertIn("fresh clone", warnings[0])

    def test_strict_mode_requires_local_original_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(PreflightError, "original MF evidence"):
                validate_mf_v2(_valid_config(), self._root(temporary), True)

    def test_legacy_alias_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = _valid_config()
            config["num_epochs"] = 100
            with self.assertRaisesRegex(PreflightError, "num_epochs"):
                validate_mf_v2(config, self._root(temporary))

    def test_required_stability_value_is_checked(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = _valid_config()
            config["optim"]["gradient_clip_norm"] = None
            with self.assertRaisesRegex(PreflightError, "gradient clip norm"):
                validate_mf_v2(config, self._root(temporary))

    def test_old_cache_cannot_be_reused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._root(temporary)
            config = _valid_config()
            config["evaluation"]["fid_reference_cache"] = "old-cache.npz"
            with self.assertRaisesRegex(PreflightError, "distinct FID"):
                validate_mf_v2(config, root)

    def test_live_v3_configs_and_cpu_derivatives_pass(self) -> None:
        verify_mf_v3(Path(__file__).resolve().parents[1])


if __name__ == "__main__":
    unittest.main()
