"""
Shared backbone network. FM and MF are both required to use a model
built by `build_backbone` from the same BackboneConfig, so parameter
count and architecture can never silently diverge between algorithms.

This is a small UNet-style conditional (on a scalar time-like input)
denoiser. It is intentionally simple and NOT tied to any diffusion /
flow-matching mathematics — it just maps (noisy_image, t) -> tensor of
the same shape. What that output is *interpreted as* (velocity, noise,
mean velocity, etc.) is entirely up to the algorithm implementation.
"""
import math
from typing import List

import torch
import torch.nn as nn

from config.config import BackboneConfig


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

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class SimpleUNet(nn.Module):
    """
    Minimal UNet: input (B, C, H, W) + scalar t in [0, 1] per sample,
    output (B, C, H, W). Purely a function approximator — no
    algorithm-specific meaning attached to its output here.
    """

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

        # Encoder
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

        # Bottleneck
        self.mid1 = ResBlock(cur_ch, cur_ch, cfg.time_embed_dim)
        self.mid2 = ResBlock(cur_ch, cur_ch, cfg.time_embed_dim)

        # Decoder
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

        h = self.out_conv(torch.nn.functional.silu(self.out_norm(h)))
        return h


def build_backbone(cfg: BackboneConfig, image_size: int = 32) -> nn.Module:
    """
    Single shared construction point. Both FM and MF (and MockAlgorithm)
    must obtain their model via this function so architecture and
    parameter count can never silently diverge.
    """
    if cfg.name == "simple_unet":
        model = SimpleUNet(cfg)
    elif cfg.name == "legacy_cifar_unet":
        # The historical name is retained for checkpoint/config identity, but
        # the frozen architecture is fully convolutional and was the shared
        # pixel-space U-Net.  It is valid for both 32x32 CIFAR-10 and 64x64
        # CelebA RGB tensors.  Keep it out of the 16x16 latent path so a legacy
        # pixel rerun cannot be mistaken for the later representation change.
        if image_size not in {32, 64} or cfg.in_channels != 3:
            raise ValueError(
                "legacy_cifar_unet is frozen for RGB pixel-space input at "
                "32x32 or 64x64; "
                f"received in_channels={cfg.in_channels}, image_size={image_size}"
            )
        from models.legacy_cifar_backbone import LegacyCifarUNet

        model = LegacyCifarUNet(cfg)
    else:
        raise ValueError(
            f"Unknown backbone: {cfg.name}. Expected 'simple_unet' or "
            "'legacy_cifar_unet'."
        )
    # Fully-convolutional backbone has no intrinsic notion of image
    # size; record the size it was configured for so algorithms that
    # need to allocate sampling noise (e.g. MockAlgorithm) can read it
    # without hard-coding a resolution.
    model._expected_image_size = image_size
    return model


def count_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
