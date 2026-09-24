"""
prepare_web_data.py
-------------------
Extract training metrics, evaluation results, and checkpoint sample paths
into a single JSON manifest for the web dashboard.

Usage (from project root):
    python scripts/prepare_web_data.py

Output:
    web/dashboard_data.json
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

METRICS_PATH = Path("results/aggregate/combined_metrics.jsonl")
SAMPLES_ROOT = Path("results/checkpoint_samples")
OUTPUT_PATH = Path("web/dashboard_data.json")

# Algorithms we care about for the dashboard
ALGORITHM_MAP = {
    "fm_cifar10": {
        "key": "fm",
        "label": "Flow Matching",
        "short": "FM",
        "color": "#5c6bc0",
        "order": 0,
    },
    "fm_lognorm_cifar10": {
        "key": "fm_lognorm",
        "label": "FM + Logit-Normal",
        "short": "FM-LN",
        "color": "#00aeb3",
        "order": 1,
    },
    "mf_cifar10": {
        "key": "mf",
        "label": "Mean Flow",
        "short": "MF",
        "color": "#ef8354",
        "order": 2,
    },
    "mf_distill_cifar10": {
        "key": "mf_distill",
        "label": "MF Distillation",
        "short": "MF-Distill",
        "color": "#e85d75",
        "order": 3,
    },
    "consistency_cifar10": {
        "key": "consistency",
        "label": "Consistency Models",
        "short": "Consistency",
        "color": "#8e44ad",
        "order": 4,
    },
    "reflow_cifar10": {
        "key": "reflow",
        "label": "Rectified Flow Reflow",
        "short": "Reflow",
        "color": "#27ae60",
        "order": 5,
    },
}


def load_metrics():
    """Load all metrics from the combined JSONL file."""
    records = []
    with METRICS_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def extract_training_loss(records):
    """Extract per-epoch training loss for each algorithm."""
    training = {}
    for rec in records:
        if rec.get("record_type") != "train_epoch":
            continue
        algo_name = rec.get("algorithm", "")
        if algo_name not in ALGORITHM_MAP:
            continue
        key = ALGORITHM_MAP[algo_name]["key"]
        epoch = rec.get("epoch")
        loss = rec.get("loss")
        if epoch is None or loss is None:
            continue
        training.setdefault(key, []).append({"epoch": epoch, "loss": round(loss, 6)})

    # Sort by epoch
    for key in training:
        training[key] = sorted(training[key], key=lambda x: x["epoch"])
    return training


def extract_eval_metrics(records):
    """Extract FID and IS evaluation metrics per algorithm/epoch/nfe."""
    evals = {}
    for rec in records:
        if rec.get("record_type") != "evaluation":
            continue
        algo_name = rec.get("algorithm", "")
        if algo_name not in ALGORITHM_MAP:
            continue
        key = ALGORITHM_MAP[algo_name]["key"]
        epoch = rec.get("epoch")
        nfe = rec.get("nfe")
        fid = rec.get("fid")
        is_mean = rec.get("is_mean")
        is_std = rec.get("is_std")
        if epoch is None or nfe is None:
            continue
        entry = {"epoch": epoch, "nfe": nfe}
        if fid is not None:
            entry["fid"] = round(fid, 2)
        if is_mean is not None:
            entry["is_mean"] = round(is_mean, 4)
        if is_std is not None:
            entry["is_std"] = round(is_std, 4)
        evals.setdefault(key, []).append(entry)

    for key in evals:
        evals[key] = sorted(evals[key], key=lambda x: (x["epoch"], x["nfe"]))
    return evals


def extract_sampling_perf(records):
    """Extract sampling performance (images/sec) per algorithm/epoch/nfe."""
    perf = {}
    for rec in records:
        if rec.get("record_type") != "sampling":
            continue
        algo_name = rec.get("algorithm", "")
        if algo_name not in ALGORITHM_MAP:
            continue
        key = ALGORITHM_MAP[algo_name]["key"]
        epoch = rec.get("epoch")
        nfe = rec.get("nfe")
        ips = rec.get("images_per_second")
        if epoch is None or nfe is None or ips is None:
            continue
        perf.setdefault(key, []).append({
            "epoch": epoch,
            "nfe": nfe,
            "images_per_second": round(ips, 1),
        })
    for key in perf:
        perf[key] = sorted(perf[key], key=lambda x: (x["epoch"], x["nfe"]))
    return perf


def scan_checkpoint_samples():
    """Scan the checkpoint_samples directory for available sample images."""
    samples = {}
    if not SAMPLES_ROOT.is_dir():
        return samples

    import re
    pattern = re.compile(r"epoch(\d+)_nfe(\d+)(?:_seed(\d+))?\.png$")

    for algo_dir in sorted(SAMPLES_ROOT.iterdir()):
        if not algo_dir.is_dir():
            continue
        key = algo_dir.name  # e.g. "fm", "fm_lognorm", "mf"
        entries = []
        for png_file in sorted(algo_dir.glob("epoch*_nfe*.png")):
            match = pattern.match(png_file.name)
            if match:
                epoch = int(match.group(1))
                nfe = int(match.group(2))
                # Path relative to project root for serving
                rel_path = str(png_file.relative_to(Path("."))).replace("\\", "/")
                entries.append({"epoch": epoch, "nfe": nfe, "path": rel_path})
        if entries:
            samples[key] = sorted(entries, key=lambda x: (x["epoch"], x["nfe"]))

    return samples


def build_manifest():
    """Build the complete dashboard data manifest."""
    records = load_metrics()

    # Build algorithm metadata
    algorithms = []
    for algo_name, info in sorted(ALGORITHM_MAP.items(), key=lambda x: x[1]["order"]):
        algorithms.append({
            "key": info["key"],
            "label": info["label"],
            "short": info["short"],
            "color": info["color"],
            "order": info["order"],
            "experiment": algo_name,
        })

    manifest = {
        "schema_version": 1,
        "algorithms": algorithms,
        "training_loss": extract_training_loss(records),
        "evaluation": extract_eval_metrics(records),
        "sampling_performance": extract_sampling_perf(records),
        "checkpoint_samples": scan_checkpoint_samples(),
    }
    return manifest


def main():
    manifest = build_manifest()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[OK] Dashboard data written to {OUTPUT_PATH}")
    print(f"     Algorithms: {[a['short'] for a in manifest['algorithms']]}")
    for key in manifest["training_loss"]:
        n = len(manifest["training_loss"][key])
        print(f"     {key}: {n} training epochs")
    for key in manifest["evaluation"]:
        n = len(manifest["evaluation"][key])
        print(f"     {key}: {n} evaluation records")
    for key in manifest["checkpoint_samples"]:
        n = len(manifest["checkpoint_samples"][key])
        print(f"     {key}: {n} sample images")


if __name__ == "__main__":
    main()
