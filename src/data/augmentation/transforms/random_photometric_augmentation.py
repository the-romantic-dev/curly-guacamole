from __future__ import annotations

from dataclasses import replace

import albumentations as A
import numpy as np

from src.data.augmentation.base import AIIJCAugmentation
from src.data.data_sample import DataSample


class RandomPhotometricAugmentation(AIIJCAugmentation):
    def __init__(self):
        super().__init__()

    def apply(self, sample: DataSample, rng: np.random.Generator | None = None) -> DataSample:
        seed = None if rng is None else int(
            rng.integers(0, np.iinfo(np.uint32).max)
        )
        transform = A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=0.12,
                contrast_limit=0.12,
                p=0.3,
            ),
            A.HueSaturationValue(
                hue_shift_limit=6,
                sat_shift_limit=12,
                val_shift_limit=8,
                p=0.2,
            )
        ], seed=seed)
        image = transform(image=sample.image)["image"]
        return replace(sample, image=image)
