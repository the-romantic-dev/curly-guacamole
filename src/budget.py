import torch
from torch.utils.flop_counter import FlopCounterMode


def count_gflops(model, size: int, *, channels: int = 3, use_valid_mask: bool = False) -> float:
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
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()  # Exclude training-only auxiliary heads and preserve BN buffers.
        with torch.no_grad(), counter:
            kwargs = ({"valid_mask": torch.ones(1, 1, size, size, device=device, dtype=torch.bool)}
                      if use_valid_mask else {})
            local_size = getattr(model, 'local_image_size', 0)
            if local_size:
                kwargs['local_input'] = torch.zeros(1, 15, local_size, local_size, device=device)
            model(torch.zeros(1, channels, size, size, device=device), **kwargs)
    finally:
        for module, training in modes:
            module.training = training
    return counter.get_total_flops() / 1e9
