from abc import ABC, abstractmethod
from collections.abc import Sequence

from torch import Tensor, nn


class Decoder(nn.Module, ABC):
    """Decoder contract: fine-to-coarse encoder features in, features + aux out.

    Subclasses expose output channels/stride and the mask head kernel size.
    The segmentation head stays in Segmenter to preserve checkpoint keys.
    """

    out_channels: int
    output_stride: int
    head_kernel_size: int = 1

    @abstractmethod
    def forward(self, features: Sequence[Tensor]) -> tuple[Tensor, Tensor | None]:
        raise NotImplementedError
