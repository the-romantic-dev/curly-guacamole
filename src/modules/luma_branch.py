"""Compact luminance evidence extracted on the model device before RGB640 reduction."""

import torch
from torch import nn
import torch.nn.functional as F


class LumaPreprocessor(nn.Module):
    """Native uint8 RGB list -> float32 luminance batch; no CPU feature maps.

    Resize directly from native geometry to the detail resolution. This retains
    the 1024 input used in the diagnostic experiment, not unlimited native detail.
    """

    def __init__(self, size):
        super().__init__()
        self.size = size
        self.register_buffer('weights', torch.tensor([.299, .587, .114]).view(3, 1, 1), persistent=False)

    @torch.no_grad()
    def forward(self, images):
        views = []
        for image in images:
            if image.ndim != 3 or image.shape[0] != 3 or image.dtype != torch.uint8:
                raise ValueError('native_rgb must contain uint8 CHW RGB images')
            if image.device != self.weights.device:
                raise ValueError('native_rgb must be on the model device')
            y = (image.float() * self.weights.float()).sum(0, keepdim=True) / 255.
            views.append(F.interpolate(y[None], (self.size, self.size), mode='bilinear', align_corners=False))
        return torch.cat(views)


class LumaBranch(nn.Module):
    """Shallow 8/16/24-channel extractor and one additive decoder fusion."""

    def __init__(self, global_channels, size, use_aux):
        super().__init__()
        self.preprocess = LumaPreprocessor(size)
        self.stem = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1, bias=False),
                                  nn.GroupNorm(4, 8), nn.GELU())
        self.fine = nn.ModuleList([nn.Conv2d(8, 8, k, padding=k//2, groups=8) for k in (3, 5)])
        self.downsample = nn.Sequential(self._downsample(8, 16), self._downsample(16, 24))
        self.project = nn.Conv2d(24, global_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(()))
        self.aux_head = nn.Conv2d(24, 1, 1) if use_aux else None

    @staticmethod
    def _downsample(input_channels, output_channels):
        return nn.Sequential(
            nn.Conv2d(input_channels, input_channels, 3, stride=2, padding=1, groups=input_channels, bias=False),
            nn.Conv2d(input_channels, output_channels, 1, bias=False),
            nn.GroupNorm(4, output_channels), nn.GELU())

    def forward(self, native_rgb, global_features):
        local = self.stem(self.preprocess(native_rgb))
        local = local + self.fine[0](local) + self.fine[1](local)
        local = self.downsample(local)
        aux = self.aux_head(local) if self.training and self.aux_head is not None else None
        compact = F.interpolate(local, global_features.shape[-2:], mode='bilinear', align_corners=False)
        return global_features + self.gamma * self.project(compact), aux
