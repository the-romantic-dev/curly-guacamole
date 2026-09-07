"""Optional spatial refinement blocks for EMCAD experiments."""

import torch
import torch.nn.functional as F
from torch import nn

from .layers import make_norm


class SpatialResidualRefinement(nn.Module):
    """Spend capacity on two dense 3x3 convolutions at the output scale."""

    def __init__(self, channels, hidden_channels, norm):
        super().__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, 3, padding=1, bias=False),
            make_norm(norm, hidden_channels), nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, 3, padding=1, bias=False),
            make_norm(norm, channels),
        )

    def forward(self, features):
        return features + self.residual(features)


class RGBLogitRefinement(nn.Module):
    """Predict a stride-2 logit correction from RGB, decoder features and mask."""

    def __init__(self, feature_channels, hidden_channels, rgb_channels, norm):
        super().__init__()
        self.rgb = nn.Sequential(
            nn.Conv2d(3, rgb_channels, 3, stride=2, padding=1, bias=False),
            make_norm(norm, rgb_channels), nn.ReLU(inplace=True),
            nn.Conv2d(rgb_channels, rgb_channels, 3, padding=1, bias=False),
            make_norm(norm, rgb_channels), nn.ReLU(inplace=True),
        )
        self.correction = nn.Sequential(
            nn.Conv2d(feature_channels + rgb_channels + 1, hidden_channels,
                      3, padding=1, bias=False),
            make_norm(norm, hidden_channels), nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
            make_norm(norm, hidden_channels), nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 1),
        )

    def forward(self, image, features, logits):
        detail = self.rgb(image)
        size = detail.shape[-2:]
        features = F.interpolate(features, size=size, mode='bilinear', align_corners=False)
        coarse = F.interpolate(logits, size=size, mode='bilinear', align_corners=False)
        return coarse + self.correction(torch.cat((detail, features, coarse), dim=1))
