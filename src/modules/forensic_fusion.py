from torch import nn

from src.forensic.dct.constants import CHANNEL_COUNT
from src.modules.forensic_branch import ForensicBranch
from src.modules.gated_fuse import GatedFuse
from src.modules.jpeg_branch import JPEGBranch


class ForensicFusion(nn.Module):
    """Строит forensic-пирамиду и сливает её с encoder-фичами."""

    FUSION_STRIDES = (8, 16, 32)

    def __init__(self, encoder_strides, encoder_channels, forensic_channels, use_aux=False, forensic_mode='maps'):
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
        self.forensic_mode = forensic_mode

        self.branch = JPEGBranch(forensic_channels) if forensic_mode == 'jpeg' else ForensicBranch(
            in_ch=CHANNEL_COUNT,
            channels=forensic_channels,
        )

        self.aux_head = nn.Conv2d(forensic_channels[0], 1, 1) if use_aux else None

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

    def forward(self, encoder_features, forensic_map, *, return_aux=False, jpeg=None):
        if self.forensic_mode == 'jpeg':
            available = [i for i, sample in enumerate(jpeg) if sample.get('available', True)]
            if not available:
                return (encoder_features, None) if return_aux else encoder_features
            if len(available) != len(jpeg):
                # Exclude PNG samples from the branch and fusion BatchNorm too.
                subset = [feature[available] for feature in encoder_features]
                fused, aux = self.forward(subset, None, return_aux=True, jpeg=[jpeg[i] for i in available])
                result = [feature.clone() for feature in encoder_features]
                for original, updated in zip(result, fused, strict=True):
                    original[available] = updated
                if not return_aux:
                    return result
                if aux is not None:
                    full_aux = aux.new_zeros((len(jpeg), *aux.shape[1:]))
                    full_aux[available] = aux
                    aux = full_aux
                return result, aux
            sizes = {s: encoder_features[self.encoder_strides.index(s)].shape[-2:] for s in self.FUSION_STRIDES}
            forensic_features = self.branch(jpeg, sizes)
        else:
            forensic_features = self.branch(forensic_map)

        for stride in self.FUSION_STRIDES:
            index = self.encoder_strides.index(stride)

            encoder_features[index] = self.fusion_blocks[str(stride)](
                encoder_features[index],
                forensic_features[stride],
            )

        if return_aux:
            aux = self.aux_head(forensic_features[8]) if self.aux_head is not None else None
            return encoder_features, aux
        return encoder_features
