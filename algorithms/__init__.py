"""
algorithms — generative-model algorithm implementations.

Each algorithm is a self-contained subclass of BaseAlgorithm that
encapsulates the training loss and the sampling procedure for one
particular generative-modelling approach.  The shared training /
sampling / evaluation pipeline interacts with algorithms exclusively
through the BaseAlgorithm interface, so algorithms are fully
interchangeable at the CLI level.

Available algorithms
--------------------
FlowMatchingAlgorithm
    Standard conditional flow matching (Lipman et al. 2022 / rectified
    flow): linear interpolation path, uniform-t sampling, Euler ODE
    integration at inference time.

FlowMatchingLognormAlgorithm
    Identical to FlowMatchingAlgorithm except that training timesteps
    are drawn from a logit-normal distribution (Esser et al. 2024,
    Stable Diffusion 3) rather than uniform.  Concentrates training on
    the intermediate-t regime where the model learns the most.

MeanFlowAlgorithm
    Mean Flow (Geng et al. 2025, arXiv:2505.13447): trains the
    *average* velocity over an interval [r, t] via the Mean Flow
    Identity and a Jacobian-vector-product (JVP).  Supports exact one-
    or few-step sampling via the displacement identity at inference.

MockAlgorithm
    Trivial smoke-test algorithm used to verify the shared pipeline
    end-to-end (checkpointing, timing, logging, evaluation) without
    requiring a real training run.
"""

from algorithms.base import BaseAlgorithm
from algorithms.flow_matching import FlowMatchingAlgorithm
from algorithms.flow_matching_lognorm import FlowMatchingLognormAlgorithm
from algorithms.mean_flow import MeanFlowAlgorithm
from algorithms.mock import MockAlgorithm

__all__ = [
    "BaseAlgorithm",
    "FlowMatchingAlgorithm",
    "FlowMatchingLognormAlgorithm",
    "MeanFlowAlgorithm",
    "MockAlgorithm",
]
