"""Local evidence extraction and feature fusion with the global EMCAD decoder."""

import torch
from torch import nn
import torch.nn.functional as F


class LocalBlock(nn.Sequential):
    """Depthwise spatial filtering followed by channel mixing; no batch statistics."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__(
            nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(8, out_channels), nn.GELU(),
        )


class LocalBranch(nn.Module):
    """A 24/48/96 pyramid, independent local supervision, and ungated global fusion.

    Fuses at local stride 4 and returns global-width features at local stride 2.
    The auxiliary prediction depends only on local evidence, never a coarse ROI.
    """

    def __init__(self, global_channels: int, use_aux: bool):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(15, 24, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 24), nn.GELU(), LocalBlock(24, 24),
        )
        self.down4 = nn.Sequential(LocalBlock(24, 48, stride=2), LocalBlock(48, 48))
        self.down8 = nn.Sequential(LocalBlock(48, 96, stride=2), LocalBlock(96, 96))
        self.local_decode = LocalBlock(48 + 96, 48)
        self.global_projection = nn.Conv2d(global_channels, 32, 1)
        self.fuse = LocalBlock(48 + 32, 32)
        self.output = nn.Sequential(LocalBlock(32 + 24, 32), nn.Conv2d(32, global_channels, 1))
        self.aux_head = nn.Conv2d(48, 1, 1) if use_aux else None

    def forward(self, local_input, global_features):
        s2 = self.stem(local_input)
        s4 = self.down4(s2)
        s8 = self.down8(s4)
        local = self.local_decode(torch.cat((s4, self._resize(s8, s4)), dim=1))
        aux = self.aux_head(local) if self.training and self.aux_head is not None else None
        global_features = self._resize(self.global_projection(global_features), local)
        fused = self.fuse(torch.cat((local, global_features), dim=1))
        output = self.output(torch.cat((self._resize(fused, s2), s2), dim=1))
        return output, aux

    @staticmethod
    def _resize(features, reference):
        return F.interpolate(features, size=reference.shape[-2:], mode='bilinear', align_corners=False)
