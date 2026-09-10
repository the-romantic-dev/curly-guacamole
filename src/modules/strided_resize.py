"""Learnable RGB reduction before the pretrained encoder's normalization."""

import torch
from torch import nn
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class StridedResize(nn.Module):
    """Raw RGB at twice the working size -> normalized three-channel input.

    The linear variant initially averages each non-overlapping 2x2 RGB block, exactly
    matching bilinear half-size sampling with align_corners=False. All 3x3
    weights and biases remain trainable, with no clamp or output activation.
    The nonlinear variant uses two randomly initialized convolutions and GELU.
    """

    def __init__(self, variant='linear'):
        super().__init__()
        if variant not in {'linear', 'nonlinear'}:
            raise ValueError('unknown resize_variant')
        self.conv = nn.Conv2d(3, 3, 3, stride=2, padding=1)
        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.bias.zero_()
            for channel in range(3):
                self.conv.weight[channel, channel, 1:, 1:] = .25
        if variant == 'nonlinear':
            # Standard trainable initialization; this variant is not initially bilinear.
            self.conv = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(16, 3, 3, stride=2, padding=1),
            )
        self.register_buffer('mean', torch.tensor(IMAGENET_DEFAULT_MEAN).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(IMAGENET_DEFAULT_STD).view(1, 3, 1, 1))

    def forward(self, image):
        if image.ndim != 4 or image.shape[1] != 3 or any(s % 2 for s in image.shape[-2:]):
            raise ValueError('strided_resize requires BCHW RGB with even spatial dimensions')
        return (self.conv(image).float() - self.mean) / self.std
