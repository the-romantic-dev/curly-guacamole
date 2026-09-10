"""Small independent DCT experiments; no image restoration objective."""
import math
from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F


def signed_log(x):
    """Compress the coefficient range without discarding sign or clipping tails."""
    return x.sign() * torch.log1p(x.abs()) / 8.0


class SignedDCTProjector(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.project = nn.Sequential(nn.Conv2d(64, channels, 1, bias=False),
                                     nn.GroupNorm(1, channels), nn.GELU())
        self.scale = nn.Parameter(torch.tensor(0.01))

    def forward(self, coefficients):
        x = F.pixel_unshuffle(coefficients[:, None].float(), 8)
        return self.scale * self.project(signed_log(x))


class SubblockDCT(nn.Module):
    """Exact change of basis: dequantized 8x8 DCT -> four 4x4 DCT blocks.

    Output channels are row-major (u, v), not zigzag; spatial positions are
    subblocks. No clipping, RGB conversion, interpolation or learned recovery.
    """
    def __init__(self):
        super().__init__()
        def basis(n):
            k = torch.arange(n, dtype=torch.float64)[:, None]
            t = torch.arange(n, dtype=torch.float64)[None]
            matrix = (2 / n) ** .5 * torch.cos(math.pi * (t + .5) * k / n)
            matrix[0] /= 2 ** .5
            return matrix
        transform = torch.block_diag(basis(4), basis(4)) @ basis(8).T
        self.register_buffer('transform', transform.float())

    def forward(self, coefficients, qtable):
        # Keep the basis conversion accurate under mixed precision.
        context = (nullcontext() if coefficients.device.type == 'meta' else
                   torch.autocast(device_type=coefficients.device.type, enabled=False))
        with context:
            b, h, w = coefficients.shape
            blocks = coefficients.float().reshape(b, h//8, 8, w//8, 8).permute(0, 1, 3, 2, 4)
            blocks = blocks * qtable.float()[:, None, None]
            converted = self.transform.float() @ blocks @ self.transform.float().T
            tiled = converted.permute(0, 1, 3, 2, 4).reshape(b, 1, h, w)
            return F.pixel_unshuffle(tiled, 4)


class SubblockProjector(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.convert = SubblockDCT()
        self.project = nn.Sequential(nn.Conv2d(16, channels, 1, bias=False),
                                     nn.GroupNorm(1, channels), nn.GELU(),
                                     nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
                                     nn.GELU())

    def forward(self, coefficients, qtable):
        return self.project(signed_log(self.convert(coefficients, qtable)))


class SpatialFrequencyBlock(nn.Module):
    """Window spatial attention plus image-conditioned attention over 64 frequencies.

    The 512 input channels are [raw, Q-weighted] x 4 stem features x 64
    frequencies. Frequency tokens retain this ordering before channel projection.
    This is a compact adaptation, not the full DCTransformer SFTB.
    """
    def __init__(self, width=32, window=4):
        super().__init__()
        self.window = window
        self.spatial_in = nn.Conv2d(512, width, 1)
        self.spatial_norm = nn.LayerNorm(width)
        self.spatial_qkv = nn.Linear(width, 3*width)
        self.spatial_out = nn.Conv2d(width, 512, 1)
        self.frequency_norm = nn.LayerNorm(8)
        self.frequency_position = nn.Parameter(torch.zeros(1, 64, 8))
        nn.init.normal_(self.frequency_position, std=.02)
        self.frequency_qkv = nn.Linear(8, 24)
        self.frequency_out = nn.Linear(8, 8)
        self.scale = nn.Parameter(torch.tensor(0.01))

    @staticmethod
    def attend(tokens, qkv, valid=None):
        q, k, v = qkv(tokens).chunk(3, dim=-1)
        scores = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(q.shape[-1])
        if valid is not None:
            scores = scores.masked_fill(~valid[:, None, :], float('-inf'))
        return (scores.softmax(-1).to(v.dtype) @ v)

    def forward(self, x):
        b, _, h, w = x.shape
        frequency = x.reshape(b, 8, 64, h, w).float().mean((-1, -2)).transpose(1, 2)
        frequency = self.frequency_norm(frequency) + self.frequency_position
        gate = self.frequency_out(self.attend(frequency, self.frequency_qkv)).tanh()
        gate = gate.transpose(1, 2).reshape(b, 512, 1, 1)
        spatial = self.spatial_in(x)
        m = self.window
        ph, pw = (-h) % m, (-w) % m
        spatial = F.pad(spatial, (0, pw, 0, ph))
        hh, ww = h+ph, w+pw
        c = spatial.shape[1]
        tokens = spatial.reshape(b, c, hh//m, m, ww//m, m).permute(0, 2, 4, 3, 5, 1).reshape(-1, m*m, c)
        valid = F.pad(torch.ones(b, h, w, device=x.device, dtype=torch.bool), (0, pw, 0, ph))
        valid = valid.reshape(b, hh//m, m, ww//m, m).permute(0, 1, 3, 2, 4).reshape(-1, m*m)
        tokens = self.attend(self.spatial_norm(tokens), self.spatial_qkv, valid)
        spatial = tokens.reshape(b, hh//m, ww//m, m, m, c).permute(0, 5, 1, 3, 2, 4).reshape(b, c, hh, ww)
        spatial = self.spatial_out(spatial[:, :, :h, :w])
        return x + self.scale * (x * gate.to(x.dtype) + spatial)
