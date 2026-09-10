"""Bilinear RGB skip with a learned correction at the target resolution.

Paper architecture: Talebi & Milanfar, ICCV 2021, arxiv.org/abs/2103.09950,
Fig. 3 (16 channels, one residual block). Layer ordering follows the public
Keras example: keras.io/examples/vision/learnable_resizer/.
Both variants use standard PyTorch random convolution initialization.
"""

from torch import nn
import torch.nn.functional as F


class ResizerResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels, eps=.001, momentum=.01),
            nn.LeakyReLU(.2),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels, eps=.001, momentum=.01),
        )

    def forward(self, features):
        return features + self.layers(features)


class ResidualImageResize(nn.Module):
    """Raw RGB -> resized RGB, without normalization or output clipping.

    Paper: high-resolution 7x7/1x1 stem, bilinear features, one residual block,
    3x3 projection plus feature skip, 7x7 RGB projection, bilinear image skip.
    Compact: 8-channel 3x3 stem, bilinear features, 3x3 hidden/RGB projections;
    no BatchNorm or internal feature skips. The RGB skip is shared by both.
    """

    def __init__(self, *, compact=False):
        super().__init__()
        self.compact = compact
        channels = 8 if compact else 16
        kernel = 3 if compact else 7
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, kernel, padding=kernel // 2),
            nn.LeakyReLU(.2),
        )
        if compact:
            self.body = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1), nn.LeakyReLU(.2),
            )
        else:
            self.stem.extend([
                nn.Conv2d(channels, channels, 1), nn.LeakyReLU(.2),
                nn.BatchNorm2d(channels, eps=.001, momentum=.01),
            ])
            self.body = nn.Sequential(
                ResizerResidualBlock(channels),
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels, eps=.001, momentum=.01),
            )
        self.output = nn.Conv2d(channels, 3, kernel, padding=kernel // 2)

    def forward(self, image):
        size = (image.shape[-2] // 2, image.shape[-1] // 2)
        baseline = F.interpolate(image.float(), size=size, mode='bilinear', align_corners=False)
        features = F.interpolate(self.stem(image), size=size, mode='bilinear', align_corners=False)
        correction = self.body(features)
        if not self.compact:
            correction = correction + features
        return baseline + self.output(correction).float()
