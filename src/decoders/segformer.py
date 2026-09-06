import torch
import torch.nn.functional as F
from torch import nn

from .layers import make_norm
from .base import Decoder
from .registry import register_decoder


@register_decoder("segformer")
class SegFormerDecoder(Decoder):
    """Project and fuse encoder scales at the finest encoder resolution.

    Pointwise convolutions implement the per-pixel linear projections of
    SegFormer's MLP head. The optional auxiliary head supervises the finest
    projected feature before fusion, independently of the main mask head.
    """

    def __init__(self, encoder_channels, embed_dim=128, norm="batch", use_aux=False, encoder_strides=(4, 8, 16, 32)):
        super().__init__()
        if not encoder_channels or embed_dim <= 0:
            raise ValueError("encoder_channels and a positive embed_dim are required")
        self.projections = nn.ModuleList(
            nn.Conv2d(channels, embed_dim, kernel_size=1)
            for channels in encoder_channels
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(len(encoder_channels) * embed_dim, embed_dim, 1, bias=False),
            make_norm(norm, embed_dim),
            nn.ReLU(inplace=True),
        )
        self.out_channels = embed_dim
        self.output_stride = encoder_strides[0]
        self.aux_head = nn.Conv2d(embed_dim, 1, 1) if use_aux else None

    def forward(self, encoder_features):
        target_size = encoder_features[0].shape[-2:]
        projected = []
        for projection, features in zip(self.projections, encoder_features, strict=True):
            features = projection(features)
            if features.shape[-2:] != target_size:
                features = F.interpolate(features, size=target_size, mode="bilinear", align_corners=False)
            projected.append(features)
        aux_logits = (
            self.aux_head(projected[0])
            if self.training and self.aux_head is not None else None
        )
        return self.fusion(torch.cat(projected, dim=1)), aux_logits
