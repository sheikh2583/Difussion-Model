"""
algorithms/r_embed.py - r-conditioning for Mean Flow algorithms.

Encodes the scalar left-endpoint r in the same sinusoidal+MLP embedding
space as the backbone's time signal. The output is added to the backbone's
t-embedding before any ResBlock sees it.

Parameter count at dimension 256: 2*(256*256 + 256) = 131,584. This is an
algorithm-specific module, separate from the shared backbone.
"""

import torch
import torch.nn as nn

from models.backbone import SinusoidalTimeEmbedding


class RCond(nn.Module):
    """Map scalar interval endpoints to the backbone time-embedding space."""

    def __init__(self, time_embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            SinusoidalTimeEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """Return an embedding of shape ``(B, time_embed_dim)``."""
        return self.net(r)


class LegacyREmbed(nn.Module):
    """Historical image-space r conditioner for existing MF checkpoints.

    It is selected only when checkpoint tensor shapes prove that the payload
    predates ``RCond``. New training continues to use ``RCond``.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, in_channels),
        )

    def forward(self, z: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        r_signal = self.net(r.unsqueeze(-1))
        return z + r_signal[:, :, None, None]


class REmbed:
    """Removed pixel-space conditioner retained only as an explicit guard."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "REmbed is removed. Use RCond(time_embed_dim) from "
            "algorithms/r_embed.py."
        )
