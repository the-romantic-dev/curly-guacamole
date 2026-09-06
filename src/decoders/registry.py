from collections.abc import Callable, Sequence
from fnmatch import fnmatchcase

from .base import Decoder


_DECODERS: dict[str, Callable[..., Decoder]] = {}


def register_decoder(name: str):
    """Register a decoder class or factory under an explicit experiment name."""
    if not isinstance(name, str) or not name or name.strip() != name:
        raise ValueError('Decoder name must be a nonempty string without surrounding whitespace')

    def register(factory):
        if name in _DECODERS:
            raise ValueError(f'Decoder {name!r} is already registered')
        _DECODERS[name] = factory
        return factory

    return register


def list_decoders(filter: str = '*') -> list[str]:
    """List registered names in alphabetical order, optionally using a glob."""
    return sorted(name for name in _DECODERS if fnmatchcase(name, filter))


def is_decoder(name: str) -> bool:
    return name in _DECODERS


def create_decoder(name: str, *, encoder_channels: Sequence[int],
                   encoder_strides: Sequence[int], norm: str = 'batch',
                   use_aux: bool = False, **kwargs) -> Decoder:
    """Build a registered decoder; unsupported options raise TypeError."""
    if not is_decoder(name):
        raise ValueError(f'Unknown decoder {name!r}. Available: {", ".join(list_decoders())}')
    channels, strides = tuple(encoder_channels), tuple(encoder_strides)
    if not channels or len(channels) != len(strides):
        raise ValueError('Encoder channels and strides must have equal nonzero length')
    if any(c <= 0 for c in channels) or any(s <= 0 for s in strides):
        raise ValueError('Encoder channels and strides must be positive')
    if any(a >= b for a, b in zip(strides, strides[1:])):
        raise ValueError('Encoder strides must be strictly increasing (fine to coarse)')
    decoder = _DECODERS[name](encoder_channels=channels, encoder_strides=strides,
                              norm=norm, use_aux=use_aux, **kwargs)
    if not isinstance(decoder, Decoder):
        raise TypeError(f'Decoder {name!r} must inherit Decoder')
    return decoder
