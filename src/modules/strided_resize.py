"""Learnable RGB reduction before the pretrained encoder's normalization."""

import torch
from torch import nn
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from src.modules.residual_resize import ResidualImageResize


class StridedResize(nn.Module):
    """Raw RGB at twice the working size -> normalized three-channel input.

    The linear variant initially averages each non-overlapping 2x2 RGB block, exactly
    matching bilinear half-size sampling with align_corners=False. All 3x3
    weights and biases remain trainable, with no clamp or output activation.
    The nonlinear variant starts at the same mapping using signed GELU pairs.
    """

    def __init__(self, variant='linear'):
        super().__init__()
        if variant not in {'linear', 'nonlinear', 'residual_paper', 'residual_compact'}:
            raise ValueError('unknown resize_variant')
        self.conv = nn.Conv2d(3, 3, 3, stride=2, padding=1)
        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.bias.zero_()
            for channel in range(3):
                self.conv.weight[channel, channel, 1:, 1:] = .25
        if variant == 'nonlinear':
            self.conv = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(16, 3, 3, stride=2, padding=1),
            )
            self._initialize_nonlinear()
        elif variant in {'residual_paper', 'residual_compact'}:
            self.conv = ResidualImageResize(compact=variant == 'residual_compact')
        self.register_buffer('mean', torch.tensor(IMAGENET_DEFAULT_MEAN).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(IMAGENET_DEFAULT_STD).view(1, 3, 1, 1))

    @torch.no_grad()
    def _initialize_nonlinear(self):
        """Use GELU(z) - GELU(-z) = z; all 16 hidden channels are active.

        Eight signed pairs share RGB (3/3/2 pairs). Distinct gains break pair
        symmetry; inverse gains and pair counts preserve the RGB average.
        """
        first, last = self.conv[0], self.conv[2]
        first.weight.zero_()
        first.bias.zero_()
        last.weight.zero_()
        last.bias.zero_()
        pair_counts = (3, 3, 2)
        for pair in range(8):
            channel = pair % 3
            gain = 1 + pair / 8
            positive, negative = 2 * pair, 2 * pair + 1
            first.weight[positive, channel, 1, 1] = gain
            first.weight[negative, channel, 1, 1] = -gain
            weight = .25 / (pair_counts[channel] * gain)
            last.weight[channel, positive, 1:, 1:] = weight
            last.weight[channel, negative, 1:, 1:] = -weight

    def forward(self, image):
        if image.ndim != 4 or image.shape[1] != 3 or any(s % 2 for s in image.shape[-2:]):
            raise ValueError('strided_resize requires BCHW RGB with even spatial dimensions')
        return (self.conv(image).float() - self.mean) / self.std
