"""
algorithms/r_embed.py — shared r-conditioning module.

Used by MeanFlowAlgorithm and MeanFlowDistillAlgorithm.
Encodes the scalar left-endpoint r into a per-channel additive bias
that is broadcast over (H, W) and added to the input image before
the backbone sees it.  This is the ONLY sanctioned extension to the
shared backbone interface (see project spec, §2).

Architecture:  Linear(1→64) → SiLU → Linear(64→C)
Output shape:  (B, C) → unsqueeze → (B, C, 1, 1) → broadcast (B, C, H, W)
"""

import torch
import torch.nn as nn


class REmbed(nn.Module):
    """Per-channel additive r-conditioning for Mean Flow algorithms."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, in_channels),
        )

    def forward(self, z: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, C, H, W) noisy image at time t.
            r: (B,) left endpoint of the Mean Flow interval.
        Returns:
            (B, C, H, W) — z with r-signal added as a spatial bias.
        """
        r_signal = self.net(r.unsqueeze(-1))          # (B, C)
        return z + r_signal[:, :, None, None]         # broadcast over H, W
