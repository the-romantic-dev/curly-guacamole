import torch.nn.functional as F
from torch import nn

from src.forensic.dct.constants import CHANNEL_COUNT, STRIDE
from src.modules.forensic_fusion import ForensicFusion
from src.modules.gate_head import GateHead
from src.decoders import create_decoder
from src.modules.utils import build_timm_encoder


class Segmenter(nn.Module):
    """Сегментатор с pretrained encoder и forensic-веткой."""

    FORENSIC_INPUT_CHANNELS = CHANNEL_COUNT

    def __init__(
        self,
        encoder_name,
        decoder_channels=(128, 64, 32, 16, 16),
        forensic_channels=(64, 96, 128),
        norm="batch",
        aux_weight=0.0,
        *,
        pretrained=True,
        use_forensics=True,
        dct_aux_weight=0.0,
        decoder_name="unet",
        decoder_embed_dim=128,
        decoder_kwargs=None,
    ):
        super().__init__()

        self.encoder, self.strides, self.channels = build_timm_encoder(
            encoder_name, pretrained=pretrained
        )

        if dct_aux_weight > 0 and not use_forensics:
            raise ValueError("DCT auxiliary head requires use_forensics")
        self.forensic_fusion = ForensicFusion(
            self.strides,
            self.channels,
            forensic_channels,
            use_aux=dct_aux_weight > 0,
        ) if use_forensics else None

        # Legacy arguments remain valid for existing notebooks and snapshots.
        options = {
            "unet": {"decoder_channels": decoder_channels},
            "segformer": {"embed_dim": decoder_embed_dim},
        }.get(decoder_name, {})
        options.update(decoder_kwargs or {})
        self.decoder = create_decoder(
            decoder_name, encoder_channels=self.channels,
            encoder_strides=self.strides, norm=norm,
            use_aux=aux_weight > 0, **options,
        )

        head_kernel = self.decoder.head_kernel_size
        self.segmentation_head = nn.Conv2d(
            self.decoder.out_channels,
            1,
            kernel_size=head_kernel,
            padding=head_kernel // 2,
        )

        self.classification_head = GateHead(
            self.channels[-1]
        )

        self.aux_weight = aux_weight

    def forensic_gate_stats(self) -> dict[str, float]:
        """Detached gate statistics for experiment logging."""
        return self.forensic_fusion.gate_stats() if self.forensic_fusion is not None else {"max_abs": 0.0}

    def forward(self, image, forensic_map=None, valid_mask=None):
        input_size = image.shape[-2:]

        encoder_features = list(
            self.encoder(image)
        )

        dct_aux_logits = None
        if self.forensic_fusion is not None:
            if forensic_map is None:
                forensic_map = self._empty_forensic_map(image)
            if self.training:
                encoder_features, dct_aux_logits = self.forensic_fusion(
                    encoder_features, forensic_map, return_aux=True,
                )
            else:
                encoder_features = self.forensic_fusion(encoder_features, forensic_map)

        decoder_features, aux_logits = self.decoder(
            encoder_features
        )

        logits = self.segmentation_head(
            decoder_features
        )

        result = {
            "logits": self._resize(logits, input_size),
            "cls_logits": self.classification_head(
                encoder_features[-1], valid_mask=valid_mask
            ),
        }

        if self.training and aux_logits is not None:
            result["aux_logits"] = self._resize(
                aux_logits,
                input_size,
            )

        if dct_aux_logits is not None:
            result["dct_aux_logits"] = dct_aux_logits
        return result

    def _empty_forensic_map(self, image):
        height, width = image.shape[-2:]

        return image.new_zeros(
            image.shape[0],
            self.FORENSIC_INPUT_CHANNELS,
            height // STRIDE,
            width // STRIDE,
        )

    @staticmethod
    def _resize(features, size):
        if features.shape[-2:] == size:
            return features

        return F.interpolate(
            features,
            size=size,
            mode="bilinear",
            align_corners=False,
        )
