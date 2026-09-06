import torch
import torch.nn.functional as F
from torch import nn

from .layers import make_norm


class UNetDecoderBlock(nn.Module):
    """Увеличивает разрешение ×2, объединяет skip-фичи и уточняет результат двумя 3×3 свёртками."""

    def __init__(
        self,
        input_channels: int,
        skip_channels: int,
        output_channels: int,
        norm: str,
    ):
        super().__init__()

        merged_channels = input_channels + skip_channels

        self.conv1 = nn.Conv2d(
            merged_channels,
            output_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.norm1 = make_norm(norm, output_channels)

        self.conv2 = nn.Conv2d(
            output_channels,
            output_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.norm2 = make_norm(norm, output_channels)

        self.activation = nn.ReLU(inplace=True)

    def forward(self, features, skip_features=None):
        features = F.interpolate(
            features,
            scale_factor=2,
            mode="nearest",
        )

        if skip_features is not None:
            features = self._match_spatial_size(
                features,
                skip_features,
            )

            features = torch.cat(
                [features, skip_features],
                dim=1,
            )

        features = self.conv1(features)
        features = self.norm1(features)
        features = self.activation(features)

        features = self.conv2(features)
        features = self.norm2(features)
        features = self.activation(features)

        return features

    @staticmethod
    def _match_spatial_size(features, reference):
        if features.shape[-2:] == reference.shape[-2:]:
            return features

        return F.interpolate(
            features,
            size=reference.shape[-2:],
            mode="nearest",
        )