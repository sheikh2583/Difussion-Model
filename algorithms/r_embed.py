"""
algorithms/r_embed.py - r-conditioning for Mean Flow algorithms.

Encodes the scalar left-endpoint r in the same sinusoidal+MLP embedding
space as the backbone's time signal. The output is added to the backbone's
t-embedding before any ResBlock sees it.

Parameter count at dimension 256: 2*(256*256 + 256) = 131,584. This is an
algorithm-specific module, separate from the shared backbone.
"""

import math

import torch
import torch.nn as nn


class _SinusoidalEmbedding(nn.Module):
    """Copy of ``models.backbone.SinusoidalTimeEmbedding``."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device).float() / half
        )
        args = t.float()[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return emb


class RCond(nn.Module):
    """Map scalar interval endpoints to the backbone time-embedding space."""

    def __init__(self, time_embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            _SinusoidalEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """Return an embedding of shape ``(B, time_embed_dim)``."""
        return self.net(r)


class REmbed:
    """Removed pixel-space conditioner retained only as an explicit guard."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "REmbed is removed. Use RCond(time_embed_dim) from "
            "algorithms/r_embed.py."
        )
