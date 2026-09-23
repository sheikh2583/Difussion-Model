"""
algorithms — all generative-model algorithm implementations.

ALGORITHM_REGISTRY maps CLI --algorithm names to classes.
To add a new algorithm: create algorithms/<name>.py, add one line here.
Trainer / Evaluator / Sampler / plots.py need zero changes.

    fm          FlowMatchingAlgorithm          — CFM, uniform-t          [Lipman 2022]
    fm_lognorm  FlowMatchingLognormAlgorithm   — CFM, logit-normal t     [Esser 2024]
    mf          MeanFlowAlgorithm              — Mean Flow, FD-JVP       [Geng 2025]
    mf_distill  MeanFlowDistillAlgorithm       — Mean Flow, FM teacher   [Geng+Salimans]
    consistency ConsistencyAlgorithm           — Consistency Models      [Song 2023]
    reflow      ReflowAlgorithm                — Rectified Flow Reflow   [Liu 2022 §3]
    mf_hutchinson MeanFlowHutchinsonAlgorithm  — MF, CV-Hutchinson VJP   [latent only]

Inference-time wrappers (not entries in ALGORITHM_REGISTRY):
    AdaptiveMeanFlowSampler      — adaptive per-sample NFE for trained MF models
    MultiScaleMeanFlowPipeline   — coarse-to-fine sampling with trained MF models

Fairness note on mf_distill and consistency:
    Both require a pre-trained FM teacher checkpoint. Their results are NOT
    directly comparable to backbone-only runs. Report teacher cost separately.
    Label plots "mf_distill" / "consistency" — never merge with "fm" or "mf".
"""

from algorithms.base import BaseAlgorithm
from algorithms.flow_matching import FlowMatchingAlgorithm
from algorithms.flow_matching_lognorm import FlowMatchingLognormAlgorithm
from algorithms.mean_flow import MeanFlowAlgorithm
from algorithms.mean_flow_hutchinson import MeanFlowHutchinsonAlgorithm
from algorithms.mean_flow_distill import MeanFlowDistillAlgorithm
from algorithms.consistency import ConsistencyAlgorithm
from algorithms.reflow import ReflowAlgorithm
from algorithms.mock import MockAlgorithm
from algorithms.mean_flow_adaptive_nfe import AdaptiveMeanFlowSampler
from algorithms.mean_flow_multiscale import MultiScaleMeanFlowPipeline

ALGORITHM_REGISTRY = {
    "fm":          FlowMatchingAlgorithm,
    "fm_lognorm":  FlowMatchingLognormAlgorithm,
    "mf":          MeanFlowAlgorithm,
    "mf_hutchinson": MeanFlowHutchinsonAlgorithm,
    "mf_distill":  MeanFlowDistillAlgorithm,
    "consistency": ConsistencyAlgorithm,
    "reflow":      ReflowAlgorithm,
    "mock":        MockAlgorithm,
}

__all__ = [
    "BaseAlgorithm",
    "FlowMatchingAlgorithm",
    "FlowMatchingLognormAlgorithm",
    "MeanFlowAlgorithm",
    "MeanFlowHutchinsonAlgorithm",
    "MeanFlowDistillAlgorithm",
    "ConsistencyAlgorithm",
    "ReflowAlgorithm",
    "MockAlgorithm",
    "AdaptiveMeanFlowSampler",
    "MultiScaleMeanFlowPipeline",
    "ALGORITHM_REGISTRY",
]
