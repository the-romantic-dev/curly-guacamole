import torch
import torch.nn.functional as F
from torch import nn


class GateHead(nn.Module):
    """Бинарная голова «в кадре есть правка», она же гейт на инференсе."""

    def __init__(self, in_channels: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(in_channels, 1)

    def forward(self, deepest, valid_mask=None):
        if valid_mask is None:
            pooled = self.pool(deepest)
        else:
            coverage = F.interpolate(valid_mask.float(), size=deepest.shape[-2:], mode="area")
            pooled = (deepest.float() * coverage).sum((2, 3), keepdim=True)
            pooled = pooled / coverage.sum((2, 3), keepdim=True).clamp_min(1e-6)
        return self.fc(self.drop(torch.flatten(pooled, 1)))
