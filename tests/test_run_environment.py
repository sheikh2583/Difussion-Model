from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from utils.results import ResultRecord, ResultsWriter
from utils.run_environment import collect_run_environment, write_run_environment
from scripts.aggregate_results import deduplicate


class _CudaUuidStub:
    """Mimic PyTorch's non-JSON-serializable private CUDA UUID value."""

    def __str__(self) -> str:
        return "GPU-deadbeef"


def test_environment_manifest_and_metrics_share_comparison_identity(tmp_path: Path) -> None:
    with (
        patch("utils.run_environment.torch.cuda.is_available", return_value=True),
        patch("utils.run_environment.torch.cuda.get_device_name", return_value="Test GPU"),
        patch(
            "utils.run_environment.torch.cuda.get_device_properties",
            return_value=type(
                "GPU",
                (),
                {"total_memory": 24 * 1024 ** 3, "uuid": _CudaUuidStub()},
            )(),
        ),
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
    assert metric["gpu_memory_gb"] == 24
    assert metric["gpu_device_index"] == 0
    assert metric["gpu_uuid"] == "GPU-deadbeef"
    assert metric["git_commit"] == "abc123"
    assert metric["git_dirty"] is False
    assert metric["code_identity"] == "abc123"


def test_machine_label_is_automatically_derived_from_host_and_gpu(tmp_path: Path) -> None:
    with (
        patch.dict("os.environ", {}, clear=True),
        patch("utils.run_environment.platform.system", return_value="Linux"),
        patch("utils.run_environment.socket.gethostname", return_value="NDAG-M-Lab"),
        patch("utils.run_environment.torch.cuda.is_available", return_value=True),
        patch(
            "utils.run_environment.torch.cuda.get_device_name",
            return_value="NVIDIA GeForce RTX 3090",
        ),
        patch(
            "utils.run_environment.torch.cuda.get_device_properties",
            return_value=type("GPU", (), {"total_memory": 24 * 1024 ** 3})(),
        ),
        patch("utils.run_environment.torch.cuda.device_count", return_value=1),
        patch("utils.run_environment._git_value", side_effect=["abc123", "", ""]),
    ):
        metadata = collect_run_environment(tmp_path)

    assert metadata["machine_label"] == (
        "linux-ndag-m-lab-nvidia-geforce-rtx-3090-24gb"
    )


def test_environment_machine_label_override_takes_precedence(tmp_path: Path) -> None:
    with (
        patch.dict("os.environ", {"DIFFUSION_MACHINE_LABEL": "my-linux-run"}, clear=True),
        patch("utils.run_environment.torch.cuda.is_available", return_value=False),
        patch("utils.run_environment.torch.cuda.device_count", return_value=0),
        patch("utils.run_environment._git_value", side_effect=["abc123", "", ""]),
    ):
        metadata = collect_run_environment(tmp_path)

    assert metadata["machine_label"] == "my-linux-run"


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
