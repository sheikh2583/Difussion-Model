"""
Machine-readable results schema shared by both algorithms. Any run
(training step, sampling run, evaluation run) is recorded as one flat
dict conforming (loosely) to this field set, then appended to a JSONL
file and optionally exported to CSV for analysis/plotting.
"""
import csv
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class ResultRecord:
    algorithm: str
    seed: int
    record_type: str  # "train_epoch" | "sampling" | "evaluation"

    epoch: Optional[int] = None
    loss: Optional[float] = None
    training_time: Optional[float] = None
    time_per_epoch: Optional[float] = None
    optimization_steps: Optional[int] = None
    samples_seen: Optional[int] = None
    parameter_count: Optional[int] = None
    trainable_parameter_count: Optional[int] = None
    algorithm_extra_parameter_count: Optional[int] = None
    peak_gpu_memory: Optional[float] = None

    nfe: Optional[int] = None
    sampling_time: Optional[float] = None
    time_per_image: Optional[float] = None
    images_per_second: Optional[float] = None

    fid: Optional[float] = None
    is_mean: Optional[float] = None
    is_std: Optional[float] = None


class ResultsWriter:
    def __init__(self, output_dir: str, experiment_name: str):
        self.jsonl_path = os.path.join(output_dir, "metrics", f"{experiment_name}.jsonl")
        os.makedirs(os.path.dirname(self.jsonl_path), exist_ok=True)

    def write(self, record: ResultRecord) -> None:
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(asdict(record)) + "\n")

    def export_csv(self, csv_path: str) -> None:
        if not os.path.exists(self.jsonl_path):
            return
        rows = []
        with open(self.jsonl_path, "r") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        if not rows:
            return
        fieldnames = list(rows[0].keys())
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
