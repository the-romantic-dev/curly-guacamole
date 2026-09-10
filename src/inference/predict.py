"""Checkpoint inference and binary masks at the original image resolution."""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np
import torch
from src.training.transfer import BatchTransfer

from src.data.letterbox import Letterbox
from src.training.builders import AmpContext


@dataclass(frozen=True)
class ThresholdConfig:
    mask_threshold: float = 0.5
    cls_threshold: float = 0.0
    min_area: float = 0.0

    def __post_init__(self) -> None:
        for name in ("mask_threshold", "cls_threshold", "min_area"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")


@dataclass(frozen=True)
class Prediction:
    image_path: str
    mask: np.ndarray


class Predictor:
    def __init__(self, model, thresholds: ThresholdConfig, amp: AmpContext) -> None:
        self.model = model.to(amp.device)
        self.thresholds = thresholds
        self.amp = amp

    def predict(self, loader: Iterable) -> Iterator[Prediction]:
        """Resize probabilities before thresholding; gate on the restored mask area."""
        self.model.eval()
        for batch in loader:
            # Leave inference/autocast contexts before yielding to caller code.
            with torch.inference_mode(), self.amp.autocast():
                kwargs = ({"valid_mask": batch["valid_mask"].to(self.amp.device)}
                          if "valid_mask" in batch else {})
                if 'jpeg' in batch:
                    kwargs['jpeg'] = BatchTransfer.move_jpeg(batch['jpeg'], self.amp.device)
                if 'local_input' in batch:
                    kwargs['local_input'] = batch['local_input'].to(self.amp.device, non_blocking=True)
                if 'native_rgb' in batch:
                    kwargs['native_rgb'] = [rgb.to(self.amp.device, non_blocking=True) for rgb in batch['native_rgb']]
                output = self.model(
                    batch["image"].to(self.amp.device),
                    batch["fmap"].to(self.amp.device) if "fmap" in batch else None,
                    **kwargs,
                )
                probabilities = output["logits"].float().sigmoid()
                cls_probs = output["cls_logits"].float().sigmoid().flatten()
            for index, image_path in enumerate(batch["image_path"]):
                size = tuple(int(value) for value in batch["original_size"][index])
                content = batch["content_size"][index] if "content_size" in batch else None
                mask = self.binary_mask(probabilities[index:index + 1], float(cls_probs[index]), size,
                                        content_size=content)
                yield Prediction(image_path, mask)

    def binary_mask(
        self, probability: torch.Tensor, cls_probability: float, size: tuple[int, int],
        *, content_size=None,
    ) -> np.ndarray:
        if len(size) != 2 or min(size) <= 0:
            raise ValueError("original size must contain positive height and width")
        restored = Letterbox.restore(probability, size, content_size)
        mask = restored[0, 0].cpu().numpy() >= self.thresholds.mask_threshold
        if cls_probability < self.thresholds.cls_threshold or mask.mean() < self.thresholds.min_area:
            mask[:] = False
        return mask.astype(np.uint8) * 255
