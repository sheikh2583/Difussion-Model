"""
models — shared neural-network backbone.

This package contains the single backbone architecture used by all
algorithms in the project.  Both Flow Matching and Mean Flow must
obtain their model through `build_backbone` so that parameter count
and architecture can never silently diverge between runs.

Key exports
-----------
build_backbone(cfg, image_size)
    Constructs and returns a backbone nn.Module from a BackboneConfig.
    All algorithms must call this function rather than instantiating
    SimpleUNet directly, so a single registry governs which architectures
    are available.

count_parameters(model)
    Returns (total_params, trainable_params) for any nn.Module.

SimpleUNet
    The minimal UNet backbone: accepts (image, t) and returns a tensor
    of the same shape.  The semantic meaning of the output (velocity,
    noise, mean velocity, etc.) is determined by the algorithm, not the
    backbone.
"""

from models.backbone import build_backbone, count_parameters, SimpleUNet

__all__ = [
    "build_backbone",
    "count_parameters",
    "SimpleUNet",
]
