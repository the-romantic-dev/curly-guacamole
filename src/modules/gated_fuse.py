import torch
import torch.nn.functional as F
from torch import nn


class GatedFuse(nn.Module):
    """Безопасно добавляет auxiliary-фичи к encoder-фичам через обучаемый поканальный gate."""

    def __init__(self, encoder_channels: int, aux_channels: int) -> None:
        super().__init__()

        self.fusion = nn.Sequential(
            nn.Conv2d(
                encoder_channels + aux_channels,
                encoder_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(encoder_channels),
            nn.GELU(),
        )

        self.channel_gate = nn.Parameter(
            torch.zeros(1, encoder_channels, 1, 1)
        )

    @staticmethod
    def _resize_like(x, reference):
        if x.shape[-2:] == reference.shape[-2:]:
            return x

        return F.interpolate(
            x,
            size=reference.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    def forward(self, encoder_features, aux_features):
        aux_features = self._resize_like(
            aux_features,
            encoder_features,
        )

        combined_features = torch.cat(
            [encoder_features, aux_features],
            dim=1,
        )

        residual = self.fusion(combined_features)
        gated_residual = self.channel_gate * residual

        return encoder_features + gated_residual
