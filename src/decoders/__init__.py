"""Pluggable segmentation decoders with a timm-style registry and factory."""

from .base import Decoder
from .registry import create_decoder, is_decoder, list_decoders, register_decoder
from .unet import UNetDecoder
from .segformer import SegFormerDecoder

__all__ = ['Decoder', 'create_decoder', 'is_decoder', 'list_decoders',
           'register_decoder', 'UNetDecoder', 'SegFormerDecoder']
