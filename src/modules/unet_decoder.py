import math

from torch import nn

from src.modules.decoder_block import UNetDecoderBlock


class UNetDecoder(nn.Module):
    """U-Net декодер с опциональной auxiliary segmentation head."""

    def __init__(
        self,
        encoder_strides,
        encoder_channels,
        decoder_channels,
        norm="batch",
        aux_stage=2,
        use_aux=False,
    ):
        super().__init__()

        self.encoder_strides = encoder_strides
        self.aux_stage = aux_stage
        self.use_aux = use_aux

        skip_channels_by_stride = dict(
            zip(encoder_strides[:-1], encoder_channels[:-1], strict=True)
        )

        num_blocks = int(math.log2(encoder_strides[-1]))

        self.blocks = nn.ModuleList()
        self.skip_strides = []

        input_channels = encoder_channels[-1]
        current_stride = encoder_strides[-1]

        for output_channels in decoder_channels[:num_blocks]:
            current_stride //= 2

            skip_channels = skip_channels_by_stride.get(
                current_stride,
                0,
            )

            self.blocks.append(
                UNetDecoderBlock(
                    input_channels,
                    skip_channels,
                    output_channels,
                    norm,
                )
            )

            self.skip_strides.append(
                current_stride if skip_channels else None
            )

            input_channels = output_channels

        self.out_channels = input_channels

        self.aux_head = (
            nn.Conv2d(
                decoder_channels[aux_stage],
                1,
                kernel_size=1,
            )
            if use_aux
            else None
        )

    def forward(self, encoder_features):
        features_by_stride = dict(
            zip(self.encoder_strides, encoder_features, strict=True)
        )

        features = encoder_features[-1]
        aux_logits = None

        for stage, (block, skip_stride) in enumerate(
            zip(self.blocks, self.skip_strides, strict=True)
        ):
            skip_features = (
                features_by_stride[skip_stride]
                if skip_stride is not None
                else None
            )

            features = block(
                features,
                skip_features,
            )

            if stage == self.aux_stage and self.aux_head is not None:
                aux_logits = self.aux_head(features)

        return features, aux_logits
