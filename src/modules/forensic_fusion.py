from torch import nn

from src.forensic.dct.constants import CHANNEL_COUNT
from src.modules.forensic_branch import ForensicBranch
from src.modules.gated_fuse import GatedFuse


class ForensicFusion(nn.Module):
    """Строит forensic-пирамиду и сливает её с encoder-фичами."""

    FUSION_STRIDES = (8, 16, 32)

    def __init__(self, encoder_strides, encoder_channels, forensic_channels):
        super().__init__()

        if len(encoder_strides) != len(encoder_channels):
            raise ValueError("encoder strides and channels must have the same length")
        if len(set(encoder_strides)) != len(encoder_strides):
            raise ValueError("encoder strides must be unique")
        if len(forensic_channels) != len(self.FUSION_STRIDES):
            raise ValueError(
                f"expected {len(self.FUSION_STRIDES)} forensic channel groups, got {len(forensic_channels)}"
            )

        missing_strides = [
            stride for stride in self.FUSION_STRIDES
            if stride not in encoder_strides
        ]
        if missing_strides:
            raise ValueError(f"encoder is missing fusion strides: {missing_strides}")

        self.encoder_strides = encoder_strides

        self.branch = ForensicBranch(
            in_ch=CHANNEL_COUNT,
            channels=forensic_channels,
        )

        self.fusion_blocks = nn.ModuleDict({
            str(stride): GatedFuse(
                encoder_channels[encoder_strides.index(stride)],
                self.branch.channels_by_stride[stride],
            )
            for stride in self.FUSION_STRIDES
        })

    def gate_stats(self) -> dict[str, float]:
        return {
            "max_abs": max(
                float(block.channel_gate.detach().abs().max())
                for block in self.fusion_blocks.values()
            )
        }

    def forward(self, encoder_features, forensic_map):
        forensic_features = self.branch(forensic_map)

        for stride in self.FUSION_STRIDES:
            index = self.encoder_strides.index(stride)

            encoder_features[index] = self.fusion_blocks[str(stride)](
                encoder_features[index],
                forensic_features[stride],
            )

        return encoder_features
