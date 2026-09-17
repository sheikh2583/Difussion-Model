#!/usr/bin/env python3
"""
Generate figures for the thesis report from archived run artefacts.

Run from project root:
    python scripts/generate_report_figures.py

The empirical figures require the three original run directories. Their default
names follow the repository's hardware-agnostic naming convention; alternate
archive directory names can be supplied through command-line options.

Output directory: docs/assets/
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "docs" / "assets"
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {
    "fm":         "#2C6FAC",
    "fm_lognorm": "#E07B30",
    "mf":         "#2E9E5B",
}
LABELS = {
    "fm":         "FM",
    "fm_lognorm": "FM + Logit-Normal",
    "mf":         "Mean Flow",
}
STYLE = {
    "figure.dpi": 150,
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
}
plt.rcParams.update(STYLE)


def load_jsonl(path: Path) -> list[dict]:
    records = []
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_all_records(results: Path, run_dirs: dict[str, str]) -> dict[str, list[dict]]:
    by_run: dict[str, list[dict]] = {}
    for run_key in LABELS:
        metrics_dir = results / run_dirs[run_key] / "metrics"
        jsonl_files = sorted(metrics_dir.glob("*.jsonl")) if metrics_dir.is_dir() else []
        records = []
        for jf in jsonl_files:
            records.extend(load_jsonl(jf))
        latest: dict[tuple, dict] = {}
        for r in records:
            key = (r.get("record_type"), r.get("epoch"), r.get("nfe"))
            latest[key] = r
        by_run[run_key] = list(latest.values())
    return by_run


def deduplicate_train(records: list[dict]) -> list[dict]:
    seen: dict[int, dict] = {}
    for r in records:
        if r.get("record_type") == "train_epoch" and r.get("epoch") is not None:
            seen[int(r["epoch"])] = r
    return [seen[e] for e in sorted(seen)]


def fig_loss_curves(by_run: dict):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for run_key, records in by_run.items():
        train = deduplicate_train(records)
        if not train:
            continue
        epochs = [r["epoch"] for r in train]
        losses = [r["loss"] for r in train]
        ax.plot(epochs, losses, color=COLORS[run_key], label=LABELS[run_key], linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss (MSE)")
    ax.set_title("Training Loss vs Epoch (audited — last duplicate per epoch)")
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT / "loss_curves.png")
    plt.close(fig)
    print("  OK loss_curves.png")


def fig_fid_vs_nfe(by_run: dict):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for run_key, records in by_run.items():
        evals = sorted(
            [r for r in records if r.get("record_type") == "evaluation" and r.get("fid") is not None],
            key=lambda r: r["nfe"],
        )
        if not evals:
            continue
        nfes = [r["nfe"] for r in evals]
        fids = [r["fid"] for r in evals]
        ax.plot(nfes, fids, marker="o", color=COLORS[run_key], label=LABELS[run_key], linewidth=2)
        for n, f in zip(nfes, fids):
            ax.annotate(f"{f:.1f}", (n, f), textcoords="offset points",
                        xytext=(4, 4), fontsize=7.5, color=COLORS[run_key])
    ax.set_xlabel("Number of Function Evaluations (NFE)")
    ax.set_ylabel("FID (lower is better)")
    ax.set_title("Sample Quality vs Inference Cost (FID)")
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT / "fid_vs_nfe.png")
    plt.close(fig)
    print("  OK fid_vs_nfe.png")


def fig_is_vs_nfe(by_run: dict):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for run_key, records in by_run.items():
        evals = sorted(
            [r for r in records if r.get("record_type") == "evaluation" and r.get("is_mean") is not None],
            key=lambda r: r["nfe"],
        )
        if not evals:
            continue
        nfes     = [r["nfe"]      for r in evals]
        is_means = [r["is_mean"]  for r in evals]
        is_stds  = [r.get("is_std") or 0 for r in evals]
        ax.errorbar(nfes, is_means, yerr=is_stds,
                    marker="o", color=COLORS[run_key], label=LABELS[run_key],
                    linewidth=2, capsize=4)
    ax.set_xlabel("Number of Function Evaluations (NFE)")
    ax.set_ylabel("Inception Score (higher is better)")
    ax.set_title("Inception Score vs Inference Cost")
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT / "is_vs_nfe.png")
    plt.close(fig)
    print("  OK is_vs_nfe.png")


def fig_quality_latency(by_run: dict):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    nfe_labels = {1: "NFE=1", 5: "NFE=5", 10: "NFE=10", 20: "NFE=20"}
    for run_key, records in by_run.items():
        fid_map  = {r["nfe"]: r["fid"]          for r in records if r.get("record_type") == "evaluation" and r.get("fid") is not None}
        time_map = {r["nfe"]: r["sampling_time"] for r in records if r.get("record_type") == "sampling"  and r.get("sampling_time") is not None}
        pts = [(time_map[n], fid_map[n], n) for n in fid_map if n in time_map]
        if not pts:
            continue
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, color=COLORS[run_key], linewidth=1.5, zorder=2)
        for t, f, n in pts:
            ax.scatter(t, f, color=COLORS[run_key], s=60, zorder=3)
            ax.annotate(nfe_labels.get(n, f"NFE={n}"), (t, f),
                        textcoords="offset points", xytext=(5, 4), fontsize=7.5,
                        color=COLORS[run_key])
    handles = [mpatches.Patch(color=COLORS[k], label=LABELS[k]) for k in LABELS]
    ax.legend(handles=handles, framealpha=0.9)
    ax.set_xlabel("Sampling time for 1,000 images (s)")
    ax.set_ylabel("FID (lower is better)")
    ax.set_title("Quality-Latency Trade-off (FID vs Wall-Clock Sampling Time)")
    fig.tight_layout()
    fig.savefig(OUT / "quality_latency.png")
    plt.close(fig)
    print("  OK quality_latency.png")


def fig_nfe_sample_matrix(results: Path, run_dirs: dict[str, str]):
    NFES = [1, 5, 20]
    run_order = ["fm", "fm_lognorm", "mf"]
    alg_prefix = {
        "fm":         "FlowMatchingAlgorithm",
        "fm_lognorm": "FlowMatchingLognormAlgorithm",
        "mf":         "MeanFlowAlgorithm",
    }
    rows, cols = len(run_order), len(NFES)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.6, rows * 3.6))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    for r_idx, run_key in enumerate(run_order):
        for c_idx, nfe in enumerate(NFES):
            ax = axes[r_idx][c_idx]
            sp = results / run_dirs[run_key] / "samples" / f"{alg_prefix[run_key]}_nfe{nfe}.png"
            if sp.exists():
                ax.imshow(Image.open(sp))
            else:
                ax.text(0.5, 0.5, f"Missing nfe{nfe}", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8)
            ax.axis("off")
            if r_idx == 0:
                ax.set_title(f"NFE = {nfe}", fontsize=11, pad=4)
        axes[r_idx][0].set_ylabel(LABELS[run_key], rotation=90, labelpad=8,
                                   fontsize=10, va="center")
        axes[r_idx][0].yaxis.set_label_position("left")
    fig.savefig(OUT / "nfe_sample_matrix.png", bbox_inches="tight")
    plt.close(fig)
    print("  OK nfe_sample_matrix.png")


def fig_final_samples_nfe20(results: Path, run_dirs: dict[str, str]):
    paths = [
        results / run_dirs["fm"]         / "samples" / "FlowMatchingAlgorithm_nfe20.png",
        results / run_dirs["fm_lognorm"] / "samples" / "FlowMatchingLognormAlgorithm_nfe20.png",
        results / run_dirs["mf"]         / "samples" / "MeanFlowAlgorithm_nfe20.png",
    ]
    labels = ["FM (epoch 100)", "FM + Logit-Normal (epoch 100)", "Mean Flow (epoch 30)"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    for ax, path, lbl in zip(axes, paths, labels):
        if Path(path).exists():
            ax.imshow(Image.open(path))
        else:
            ax.text(0.5, 0.5, "Missing", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(lbl, fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "final_samples_nfe20.png", bbox_inches="tight")
    plt.close(fig)
    print("  OK final_samples_nfe20.png")


def fig_lognorm_progression(results: Path):
    ckpt_dir = results / "checkpoint_samples" / "fm_lognorm"
    epochs = [10, 20, 30, 50, 70, 90, 100]
    selected = [(ep, ckpt_dir / f"epoch{ep:03d}_nfe20.png") for ep in epochs
                if (ckpt_dir / f"epoch{ep:03d}_nfe20.png").exists()]
    if not selected:
        print("  -- FM-LN progression: no epoch checkpoint images found, skipping.")
        return
    n = len(selected)
    fig, axes = plt.subplots(1, n, figsize=(n * 2.4, 2.8))
    if n == 1:
        axes = [axes]
    for ax, (ep, path) in zip(axes, selected):
        ax.imshow(Image.open(path))
        ax.set_title(f"Ep {ep}", fontsize=9)
        ax.axis("off")
    fig.suptitle("FM + Logit-Normal: Epoch Progression at NFE=20 (fixed seed/grid)",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "lognorm_progression.png", bbox_inches="tight")
    plt.close(fig)
    print("  OK lognorm_progression.png")


def fig_architecture():
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(x, y, w, h, label, sub="", color="#D6E4F0", fc="#2C6FAC"):
        rect = mpatches.FancyBboxPatch((x, y), w, h,
            boxstyle="round,pad=0.08", linewidth=1.5,
            edgecolor=fc, facecolor=color, zorder=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2 + (0.15 if sub else 0), label,
                ha="center", va="center", fontsize=9, fontweight="bold", color="#1a1a1a", zorder=3)
        if sub:
            ax.text(x + w/2, y + h/2 - 0.27, sub,
                    ha="center", va="center", fontsize=7.5, color="#444", zorder=3)

    def arrow(x1, y, x2):
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="->", color="#555", lw=1.3), zorder=4)

    box(0.1, 2.3, 1.5, 1.4, "Input", "3x32x32")
    arrow(1.6, 3.0, 2.1)
    for i, (ch, sz) in enumerate([(64, "32x32"), (128, "16x16"), (128, "8x8")]):
        box(2.1 + i*2.0, 2.3, 1.8, 1.4, f"Enc {i+1}", f"{ch}ch {sz}",
            color="#E8F4FD")
        if i < 2:
            arrow(2.1 + i*2.0 + 1.8, 3.0, 2.1 + (i+1)*2.0)

    box(0.1, 0.3, 2.5, 1.2, "Time Embed", "sin -> MLP 256D", color="#FFF3E0", fc="#E07B30")
    ax.annotate("", xy=(3.1, 2.3), xytext=(1.35, 1.5),
                arrowprops=dict(arrowstyle="->", color="#E07B30", lw=1.2,
                                connectionstyle="arc3,rad=-0.25"), zorder=4)

    box(8.1, 2.3, 1.8, 1.4, "Bottleneck", "128ch 8x8", color="#E8F8E8", fc="#2E9E5B")
    arrow(8.1, 3.0, 8.1)
    arrow(9.9, 3.0, 10.3)

    for i, (ch, sz) in enumerate([(128, "16x16"), (64, "32x32"), (64, "32x32")]):
        box(10.3 + i*1.5, 2.3, 1.4, 1.4, f"Dec {i+1}", f"{ch}ch {sz}",
            color="#DAEEFA")
        if i < 2:
            arrow(10.3 + i*1.5 + 1.4, 3.0, 10.3 + (i+1)*1.5)

    arrow(10.3 + 3*1.5, 3.0, 14.5)
    box(14.5, 2.3, 1.8, 1.4, "Output", "3x32x32", color="#F5E6FF", fc="#7B3FA5")

    box(0.1, 4.3, 2.8, 1.2, "r-Embed (MF only)", "Linear->SiLU->Linear 323p",
        color="#FFF9E0", fc="#B8860B")
    ax.annotate("", xy=(2.1, 3.7), xytext=(1.5, 4.3),
                arrowprops=dict(arrowstyle="->", color="#B8860B", lw=1.2,
                                connectionstyle="arc3,rad=0.3"), zorder=4)
    ax.text(3.0, 4.1, "add to x before\nfirst encoder", fontsize=7.5,
            color="#B8860B", ha="left")

    ax.set_title("SimpleUNet Backbone: 6,352,899 params  |  MF adds 323-param r-embedding",
                 fontsize=11, pad=10)
    fig.tight_layout()
    fig.savefig(OUT / "architecture.png", bbox_inches="tight", dpi=160)
    plt.close(fig)
    print("  OK architecture.png")


def fig_cifar10_examples():
    try:
        import torch
        import torchvision
        import torchvision.transforms as T
        from torchvision.utils import make_grid
        tf = T.Compose([T.ToTensor(), T.Normalize((0.5,)*3, (0.5,)*3)])
        ds = torchvision.datasets.CIFAR10(root=str(ROOT / "data" / "raw"),
                                           train=True, transform=tf, download=False)
        imgs = torch.stack([ds[i][0] for i in range(16)])
        grid = make_grid(imgs, nrow=4, padding=2, normalize=True, value_range=(-1, 1))
        np_grid = (grid.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype("uint8")
        Image.fromarray(np_grid).save(OUT / "cifar10_examples.png")
        print("  OK cifar10_examples.png  (from dataset)")
    except Exception as exc:
        existing = OUT / "cifar10_examples.png"
        if existing.exists():
            print(f"  -- kept bundled cifar10_examples.png ({exc})")
            return
        raise RuntimeError(
            "CIFAR-10 is unavailable. Run bootstrap.py --datasets cifar10 first."
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--fm-run", default="fm_cifar10")
    parser.add_argument("--fm-lognorm-run", default="fm_lognorm_cifar10")
    parser.add_argument("--mf-run", default="mf_cifar10")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(ROOT))
    print(f"Generating figures -> {OUT}")
    run_dirs = {
        "fm": args.fm_run,
        "fm_lognorm": args.fm_lognorm_run,
        "mf": args.mf_run,
    }
    by_run = load_all_records(args.results_dir, run_dirs)
    if all(by_run.values()):
        fig_loss_curves(by_run)
        fig_fid_vs_nfe(by_run)
        fig_is_vs_nfe(by_run)
        fig_quality_latency(by_run)
        fig_nfe_sample_matrix(args.results_dir, run_dirs)
        fig_final_samples_nfe20(args.results_dir, run_dirs)
        fig_lognorm_progression(args.results_dir)
    else:
        missing = [run_dirs[key] for key, records in by_run.items() if not records]
        print("  -- empirical figures kept; missing archived metrics for: " + ", ".join(missing))
    fig_architecture()
    fig_cifar10_examples()
    print("\nDone. All figures in", OUT)


if __name__ == "__main__":
    main()
