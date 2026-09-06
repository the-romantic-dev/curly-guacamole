from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from src.data.data_sample import DataSample


@dataclass(frozen=True)
class AugmentationConfig:
    crop_scale_range: tuple[float, float]
    jpeg_recompression_probability: float
    jpeg_recompression_quality_range: tuple[int, int]
    full_frame: bool
    full_frame_probability: float = 0.0
    foreground_crop_probability: float = 0.0
    final_full_frame_epochs: int = 0

    def __post_init__(self):
        for name in ("full_frame_probability", "foreground_crop_probability"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"augmentation.{name} must be in [0, 1]")
        if self.final_full_frame_epochs < 0:
            raise ValueError("augmentation.final_full_frame_epochs must be non-negative")


class AugmentationStage(StrEnum):
    BEFORE_FORENSICS = "before_forensics"
    AFTER_FORENSICS = "after_forensics"
    FINAL = "final"


class AIIJCAugmentation(ABC):
    @abstractmethod
    def apply(self, sample: DataSample, rng: np.random.Generator | None = None) -> DataSample: ...


def require_rng(
        rng: np.random.Generator | None,
        stage: AugmentationStage,
) -> np.random.Generator:
    if rng is None:
        raise ValueError(f"{stage.value} augmentations require rng")
    return rng


def require_fmap(sample: DataSample) -> np.ndarray:
    if sample.fmap is None:
        raise ValueError("forensic map is required for this augmentation stage")
    return sample.fmap
