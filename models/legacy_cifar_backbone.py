"""Frozen pixel backbone used by the original checkpoint-backed runs.

This module intentionally preserves the historical ``models/backbone.py``
architecture whose Git blob is ``d1b02c84feba4f1f4360a7adf54c3a98729824c8``.
The historical public name remains ``legacy_cifar_unet``, but the network is
fully convolutional and supports the project's 32x32 CIFAR-10 and 64x64 CelebA
pixel spaces. Keep its layer construction and state-dict keys stable.
"""

import math
from typing import List

import torch
import torch.nn as nn

from config.config import BackboneConfig


LEGACY_BACKBONE_NAME = "legacy_cifar_unet"
LEGACY_BACKBONE_GIT_BLOB = "d1b02c84feba4f1f4360a7adf54c3a98729824c8"


class SinusoidalTimeEmbedding(nn.Module):
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


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_embed_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_embed_dim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(torch.nn.functional.silu(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(torch.nn.functional.silu(self.norm2(h)))
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class LegacyCifarUNet(nn.Module):
    """Exact trainable pixel architecture used by the early project runs."""

    def __init__(self, cfg: BackboneConfig):
        super().__init__()
        self.cfg = cfg
        ch = cfg.base_channels
        mults: List[int] = cfg.channel_mults

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(cfg.time_embed_dim),
            nn.Linear(cfg.time_embed_dim, cfg.time_embed_dim),
            nn.SiLU(),
            nn.Linear(cfg.time_embed_dim, cfg.time_embed_dim),
        )
        self.in_conv = nn.Conv2d(cfg.in_channels, ch, 3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        chans = [ch]
        cur_ch = ch
        for i, mult in enumerate(mults):
            out_ch = cfg.base_channels * mult
            stage = nn.ModuleList()
            for _ in range(cfg.num_res_blocks):
                stage.append(ResBlock(cur_ch, out_ch, cfg.time_embed_dim))
                cur_ch = out_ch
                chans.append(cur_ch)
            self.down_blocks.append(stage)
            if i != len(mults) - 1:
                self.downsamples.append(Downsample(cur_ch))
                chans.append(cur_ch)
            else:
                self.downsamples.append(nn.Identity())

        self.mid1 = ResBlock(cur_ch, cur_ch, cfg.time_embed_dim)
        self.mid2 = ResBlock(cur_ch, cur_ch, cfg.time_embed_dim)

        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for i, mult in reversed(list(enumerate(mults))):
            out_ch = cfg.base_channels * mult
            stage = nn.ModuleList()
            for _ in range(cfg.num_res_blocks + 1):
                skip_ch = chans.pop()
                stage.append(ResBlock(cur_ch + skip_ch, out_ch, cfg.time_embed_dim))
                cur_ch = out_ch
            self.up_blocks.append(stage)
            if i != 0:
                self.upsamples.append(Upsample(cur_ch))
            else:
                self.upsamples.append(nn.Identity())

        self.out_norm = nn.GroupNorm(8, cur_ch)
        self.out_conv = nn.Conv2d(cur_ch, cfg.in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)
        h = self.in_conv(x)
        skips = [h]

        for stage, down in zip(self.down_blocks, self.downsamples):
            for block in stage:
                h = block(h, t_emb)
                skips.append(h)
            h = down(h)
            if not isinstance(down, nn.Identity):
                skips.append(h)

        h = self.mid1(h, t_emb)
        h = self.mid2(h, t_emb)

        for stage, up in zip(self.up_blocks, self.upsamples):
            for block in stage:
                skip = skips.pop()
                h = block(torch.cat([h, skip], dim=1), t_emb)
            h = up(h)

        return self.out_conv(torch.nn.functional.silu(self.out_norm(h)))
