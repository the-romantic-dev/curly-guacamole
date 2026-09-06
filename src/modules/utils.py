import timm
from torch import nn


from src.decoders.layers import make_norm


def build_timm_encoder(name: str, *, pretrained: bool = True):
    encoder = timm.create_model(name, features_only=True, pretrained=pretrained, in_chans=3)
    reductions = list(encoder.feature_info.reduction())
    channels = list(encoder.feature_info.channels())
    return encoder, reductions, channels


def init_module(module: nn.Module) -> None:
    """Kaiming под ReLU/GELU для всего, что учится с нуля."""
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
