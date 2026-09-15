"""
evaluation — FID / Inception Score evaluation pipeline.

This package contains two public components:

Evaluator
    High-level evaluation orchestrator.  For each NFE value it calls
    Sampler.generate_for_evaluation() to obtain a batch of generated
    images, then passes them to the metric functions in metrics.py
    (FID, Inception Score) and writes a ResultRecord (evaluation record
    type) to the JSONL results log.  The Evaluator does not know which
    algorithm produced the images.

ensure_fid_reference(cfg, real_loader, device)
    Computes (once) and caches the real-image reference tensor that FID
    needs.  Subsequent runs reuse the cached tensor after verifying that
    its metadata (image count, resolution, channels) matches the current
    config.  This guarantees every algorithm is evaluated against the
    same reference distribution.

metrics.py (internal)
    Low-level functions compute_fid() and compute_inception_score()
    built on top of torchmetrics.  Wrapped to feed images in fixed-
    size batches so VRAM usage is bounded regardless of sample count.
"""

from evaluation.evaluator import Evaluator, ensure_fid_reference

__all__ = [
    "Evaluator",
    "ensure_fid_reference",
]
