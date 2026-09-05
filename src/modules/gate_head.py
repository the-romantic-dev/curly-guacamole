import torch
from torch import nn


class GateHead(nn.Module):
    """Бинарная голова «в кадре есть правка», она же гейт на инференсе."""

    def __init__(self, in_channels: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(in_channels, 1)

    def forward(self, deepest):
        return self.fc(self.drop(torch.flatten(self.pool(deepest), 1)))