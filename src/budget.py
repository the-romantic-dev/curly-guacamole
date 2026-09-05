import torch
from torch.utils.flop_counter import FlopCounterMode


def count_gflops(model, size: int, *, channels: int = 3) -> float:
    """Строгие GFLOPs одного forward на входе (1, channels, size, size).

    Модель считается там, где лежит: перекладывать её здесь нельзя, иначе
    вызывающий получил бы обратно испорченный объект. На `meta`-устройстве
    работает и даёт то же число, но без арифметики и без памяти — доли секунды
    против нескольких секунд на 768px.
    """

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    counter = FlopCounterMode(display=False)
    with torch.no_grad(), counter:
        model(torch.zeros(1, channels, size, size, device=device))
    return counter.get_total_flops() / 1e9
