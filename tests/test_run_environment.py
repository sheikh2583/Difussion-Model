from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from utils.results import ResultRecord, ResultsWriter
from utils.run_environment import collect_run_environment, write_run_environment
from scripts.aggregate_results import deduplicate


def test_environment_manifest_and_metrics_share_comparison_identity(tmp_path: Path) -> None:
    with (
        patch("utils.run_environment.torch.cuda.is_available", return_value=True),
        patch("utils.run_environment.torch.cuda.get_device_name", return_value="Test GPU"),
        patch("utils.run_environment.torch.cuda.device_count", return_value=1),
        patch("utils.run_environment._git_value", side_effect=["abc123", "", ""]),
    ):
        metadata = collect_run_environment(tmp_path, machine_label="linux-lab")

    write_run_environment(tmp_path, metadata)
    writer = ResultsWriter(str(tmp_path), "test_run")
    writer.write(ResultRecord(algorithm="Algo", seed=0, record_type="train_epoch"))

    latest = json.loads((tmp_path / "run_environment.json").read_text())
    metric = json.loads(writer.jsonl_path and Path(writer.jsonl_path).read_text())
    assert latest["session_id"] == metric["session_id"]
    assert metric["machine_label"] == "linux-lab"
    assert metric["gpu_name"] == "Test GPU"
    assert metric["git_commit"] == "abc123"
    assert metric["git_dirty"] is False
    assert metric["code_identity"] == "abc123"


def test_environment_history_appends_sessions(tmp_path: Path) -> None:
    first = {"session_id": "one", "machine_label": "windows-lab"}
    second = {"session_id": "two", "machine_label": "linux-lab"}
    write_run_environment(tmp_path, first)
    write_run_environment(tmp_path, second)

    history = [
        json.loads(line)
        for line in (tmp_path / "run_environment.jsonl").read_text().splitlines()
    ]
    assert history == [first, second]
    assert json.loads((tmp_path / "run_environment.json").read_text()) == second


def test_aggregation_does_not_merge_machines_or_code_revisions() -> None:
    base = {
        "experiment": "fm_cifar10",
        "seed": 0,
        "record_type": "train_epoch",
        "epoch": 10,
        "nfe": None,
    }
    records = [
        {**base, "machine_label": "linux", "code_identity": "aaa"},
        {**base, "machine_label": "windows", "code_identity": "aaa"},
        {**base, "machine_label": "linux", "code_identity": "bbb"},
    ]
    assert len(deduplicate(records)) == 3
