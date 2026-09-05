from __future__ import annotations

from dataclasses import replace

import numpy as np

from src.data.augmentation.base import AIIJCAugmentation, AugmentationStage, require_fmap, require_rng
from src.data.data_sample import DataSample
from src.data.utils import align8
from src.forensic.dct import crop_fmaps


class RandomDCTAlignedCrop(AIIJCAugmentation):
    def __init__(self, crop_scale_range: tuple[float, float], full_frame: bool):
        super().__init__()
        self.crop_scale_range = crop_scale_range
        self.full_frame = full_frame

    def sample_crop(
            self,
            height: int,
            width: int,
            rng: np.random.Generator,
    ) -> tuple[int, int, int]:
        """Sample square crop with size and coordinates divisible by 8."""
        max_side = align8(min(height, width))
        if max_side < 8:
            raise ValueError(f"image {height}x{width} is smaller than 8x8")

        side = int(round(np.sqrt(rng.uniform(*self.crop_scale_range)) * min(height, width)))
        side = min(align8(max(64, side)), max_side)

        top = align8(int(rng.integers(0, height - side + 1)))
        left = align8(int(rng.integers(0, width - side + 1)))
        return top, left, side

    def apply(self, sample: DataSample, rng: np.random.Generator | None = None) -> DataSample:
        if self.full_frame:
            return sample

        rng = require_rng(rng, AugmentationStage.AFTER_FORENSICS)
        fmap = require_fmap(sample)
        height, width = sample.image.shape[:2]
        top, left, side = self.sample_crop(height, width, rng)

        image = sample.image[top:top + side, left:left + side]
        fmap = crop_fmaps(fmap, top, left, side, side)
        mask = (
            sample.mask[top:top + side, left:left + side]
            if sample.mask is not None
            else None
        )

        return replace(sample, image=image, mask=mask, fmap=fmap)
