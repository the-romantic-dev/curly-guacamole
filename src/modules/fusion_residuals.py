"""Compact RGB/JPEG fusion residuals, evaluated on the aligned encoder grid."""

import torch
from torch import nn
from torch.nn import functional as F


class LocalFusionResidual(nn.Module):
    """Shared local residual; optional content-dependent spatial/channel gate."""

    def __init__(self, encoder_channels, aux_channels, variant, width=32):
        super().__init__()
        self.rgb = nn.Conv2d(encoder_channels, width, 1)
        self.jpeg = nn.Conv2d(aux_channels, width, 1)
        self.context = nn.Sequential(
            nn.Conv2d(2 * width, 2 * width, 3, padding=1, groups=2 * width),
            nn.GELU())
        self.output = nn.Conv2d(2 * width, encoder_channels, 1)
        gate_channels = 1 if variant == 'spatial' else encoder_channels
        self.gate = (nn.Conv2d(2 * width, gate_channels, 1)
                     if variant != 'local' else None)

    def forward(self, rgb, jpeg):
        context = self.context(torch.cat((self.rgb(rgb), self.jpeg(jpeg)), dim=1))
        residual = self.output(context)
        return residual if self.gate is None else residual * self.gate(context).sigmoid()


class FiLMFusionResidual(nn.Module):
    """JPEG-conditioned spatial affine modulation of RGB features."""

    def __init__(self, encoder_channels, aux_channels, width=32):
        super().__init__()
        self.modulation = nn.Sequential(
            nn.Conv2d(aux_channels, width, 1), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1, groups=width), nn.GELU(),
            nn.Conv2d(width, 2 * encoder_channels, 1))

    def forward(self, rgb, jpeg):
        gamma, beta = self.modulation(jpeg).chunk(2, dim=1)
        return gamma.tanh() * rgb + beta


class WindowCrossAttentionResidual(nn.Module):
    """RGB queries read JPEG keys/values inside nonoverlapping 4x4 windows."""

    def __init__(self, encoder_channels, aux_channels, width=32, window=4):
        super().__init__()
        self.window = window
        self.query = nn.Conv2d(encoder_channels, width, 1)
        self.key_value = nn.Conv2d(aux_channels, 2 * width, 1)
        self.query_norm = nn.LayerNorm(width)
        self.key_norm = nn.LayerNorm(width)
        self.output = nn.Conv2d(width, encoder_channels, 1)

    def partition(self, x):
        b, c, h, w = x.shape
        m = self.window
        return x.reshape(b, c, h // m, m, w // m, m).permute(
            0, 2, 4, 3, 5, 1).reshape(-1, m * m, c)

    def forward(self, rgb, jpeg):
        b, _, h, w = rgb.shape
        m = self.window
        padding = (0, (-w) % m, 0, (-h) % m)
        query = F.pad(self.query(rgb), padding)
        key, value = F.pad(self.key_value(jpeg), padding).chunk(2, dim=1)
        hh, ww = query.shape[-2:]
        q = self.query_norm(self.partition(query))
        k = self.key_norm(self.partition(key))
        v = self.partition(value)
        valid = F.pad(torch.ones(b, 1, h, w, device=rgb.device, dtype=torch.bool), padding)
        valid = self.partition(valid).squeeze(-1)
        # Explicit matmuls also remain visible to the competition FLOP counter.
        scores = (q.float() @ k.float().transpose(-1, -2)) / q.shape[-1] ** 0.5
        scores = scores.masked_fill(~valid[:, None, :], float('-inf'))
        attended = scores.softmax(-1).to(v.dtype) @ v
        spatial = attended.reshape(b, hh // m, ww // m, m, m, v.shape[-1])
        spatial = spatial.permute(0, 5, 1, 3, 2, 4).reshape(b, v.shape[-1], hh, ww)
        return self.output(spatial[:, :, :h, :w])
