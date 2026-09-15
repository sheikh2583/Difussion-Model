"""
Reusable plotting utilities operating purely on the JSONL/CSV results
schema (utils/results.py). No knowledge of FM/MF internals is needed —
plots are grouped by the "algorithm" field only.
"""
import json
import os
from collections import defaultdict
from typing import List

import matplotlib.pyplot as plt


def _load_records(jsonl_path: str) -> List[dict]:
    records = []
    if not os.path.exists(jsonl_path):
        return records
    with open(jsonl_path, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def _group_by_algorithm(records: List[dict]) -> dict:
    grouped = defaultdict(list)
    for r in records:
        grouped[r["algorithm"]].append(r)
    return grouped


def plot_loss_vs_epoch(jsonl_path: str, out_path: str) -> None:
    records = [r for r in _load_records(jsonl_path) if r["record_type"] == "train_epoch"]
    grouped = _group_by_algorithm(records)

    plt.figure()
    for algo, rows in grouped.items():
        rows = sorted(rows, key=lambda r: r["epoch"])
        plt.plot([r["epoch"] for r in rows], [r["loss"] for r in rows], label=algo)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss vs Epoch")
    plt.legend()
    _save(out_path)


def plot_training_time_comparison(jsonl_path: str, out_path: str) -> None:
    records = [r for r in _load_records(jsonl_path) if r["record_type"] == "train_epoch"]
    grouped = _group_by_algorithm(records)

    algos, totals = [], []
    for algo, rows in grouped.items():
        algos.append(algo)
        totals.append(sum(r["time_per_epoch"] for r in rows if r["time_per_epoch"]))

    plt.figure()
    plt.bar(algos, totals)
    plt.ylabel("Total Training Time (s)")
    plt.title("Training Time Comparison")
    _save(out_path)


def plot_sampling_time_vs_nfe(jsonl_path: str, out_path: str) -> None:
    _plot_metric_vs_nfe(jsonl_path, "sampling", "sampling_time",
                         "Sampling Time (s)", "Sampling Time vs NFE", out_path)


def plot_fid_vs_nfe(jsonl_path: str, out_path: str) -> None:
    _plot_metric_vs_nfe(jsonl_path, "evaluation", "fid",
                         "FID", "FID vs NFE", out_path)


def plot_is_vs_nfe(jsonl_path: str, out_path: str) -> None:
    _plot_metric_vs_nfe(jsonl_path, "evaluation", "is_mean",
                         "Inception Score", "IS vs NFE", out_path)


def plot_gpu_memory_comparison(jsonl_path: str, out_path: str) -> None:
    records = [r for r in _load_records(jsonl_path) if r["record_type"] == "train_epoch"]
    grouped = _group_by_algorithm(records)

    algos, peaks = [], []
    for algo, rows in grouped.items():
        algos.append(algo)
        vals = [r["peak_gpu_memory"] for r in rows if r["peak_gpu_memory"] is not None]
        peaks.append(max(vals) if vals else 0)

    plt.figure()
    plt.bar(algos, peaks)
    plt.ylabel("Peak GPU Memory (MB)")
    plt.title("GPU Memory Comparison")
    _save(out_path)


def plot_fid_vs_sampling_time(jsonl_path: str, out_path: str) -> None:
    """
    Quality-vs-compute trade-off: merges evaluation FID records with
    sampling-time records at matching (algorithm, nfe) pairs.
    """
    records = _load_records(jsonl_path)
    fid_by_key = {(r["algorithm"], r["nfe"]): r["fid"]
                  for r in records if r["record_type"] == "evaluation" and r.get("fid") is not None}
    time_by_key = {(r["algorithm"], r["nfe"]): r["sampling_time"]
                   for r in records if r["record_type"] == "sampling"}

    grouped = defaultdict(list)
    for key, fid in fid_by_key.items():
        if key in time_by_key:
            grouped[key[0]].append((time_by_key[key], fid))

    plt.figure()
    for algo, pairs in grouped.items():
        pairs = sorted(pairs)
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        plt.plot(xs, ys, marker="o", label=algo)
    plt.xlabel("Sampling Time (s)")
    plt.ylabel("FID")
    plt.title("FID vs Sampling Time (Quality vs Compute Budget)")
    plt.legend()
    _save(out_path)


def _plot_metric_vs_nfe(jsonl_path: str, record_type: str, field: str,
                         ylabel: str, title: str, out_path: str) -> None:
    records = [r for r in _load_records(jsonl_path)
               if r["record_type"] == record_type and r.get(field) is not None]
    grouped = _group_by_algorithm(records)

    plt.figure()
    for algo, rows in grouped.items():
        rows = sorted(rows, key=lambda r: r["nfe"])
        plt.plot([r["nfe"] for r in rows], [r[field] for r in rows], marker="o", label=algo)
    plt.xlabel("NFE")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    _save(out_path)


def _save(out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
