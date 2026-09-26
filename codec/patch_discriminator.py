"""70×70 PatchGAN discriminator for VAE adversarial training.

Operates on 64×64 RGB inputs. Output is a spatial grid of real/fake
logits — each patch independently judged. Hinge loss is used (more
stable than BCE for image generation).
"""
import torch
import torch.nn as nn


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()
        def block(in_ch, out_ch, norm=True):
            layers = [nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=not norm)]
            if norm:
                layers.append(nn.InstanceNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        B = base_channels
        self.net = nn.Sequential(
            *block(in_channels, B,    norm=False),   # 64→32
            *block(B,           B*2),                # 32→16
            *block(B*2,         B*4),                # 16→8
            nn.Conv2d(B*4, 1, 4, 1, 1),             # 8→7, patch logits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def hinge_loss_disc(logits_real, logits_fake):
    """Discriminator hinge loss."""
    loss_real = torch.mean(torch.relu(1.0 - logits_real))
    loss_fake = torch.mean(torch.relu(1.0 + logits_fake))
    return 0.5 * (loss_real + loss_fake)


def hinge_loss_gen(logits_fake):
    """Generator (VAE) adversarial loss — fool the discriminator."""
    return -torch.mean(logits_fake)