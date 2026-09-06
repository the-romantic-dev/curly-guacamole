from torch import nn


def make_norm(kind: str, channels: int) -> nn.Module:
    """BatchNorm или GroupNorm.
    BatchNorm копит бегущие статистики внутри forward, и одно
    переполнение fp16 портит их навсегда: GradScaler защищает веса, но не буферы.
    """
    if kind == "batch":
        return nn.BatchNorm2d(channels)
    if kind == "group":
        return nn.GroupNorm(num_groups=min(32, channels), num_channels=channels)
    raise ValueError(f"неизвестная нормализация: {kind}")


