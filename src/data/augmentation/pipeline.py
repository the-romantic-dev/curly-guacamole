from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from src.data.augmentation.base import AIIJCAugmentation, AugmentationConfig, AugmentationStage
from src.data.augmentation.transforms import RandomDCTAlignedCrop
from src.data.augmentation.transforms.random_jpeg_recompression import RandomJPEGRecompression
from src.data.augmentation.transforms.random_photometric_augmentation import RandomPhotometricAugmentation
from src.data.augmentation.transforms.random_rotate_flip import RandomRotateFlip
from src.data.data_sample import DataSample


class AugmentationPipeline:
    """Stage-based synchronous augmentations for image/mask/forensic-map."""

    def __init__(
            self,
            config: AugmentationConfig | Mapping[str, Any],
    ):
        self.config = (
            config
            if isinstance(config, AugmentationConfig)
            else AugmentationConfig(**dict(config))
        )
        self.pipeline: dict[AugmentationStage, list[AIIJCAugmentation]] = {
            AugmentationStage.BEFORE_FORENSICS: [
                RandomJPEGRecompression(
                    quality_range=self.config.jpeg_recompression_quality_range,
                    probability=self.config.jpeg_recompression_probability
                )
            ],
            AugmentationStage.AFTER_FORENSICS: [
                RandomDCTAlignedCrop(crop_scale_range=self.config.crop_scale_range, full_frame=self.config.full_frame),
                RandomRotateFlip(full_frame=self.config.full_frame)
            ],
            AugmentationStage.FINAL: [
                RandomPhotometricAugmentation()
            ],
        }

    def apply(
            self,
            stage: AugmentationStage,
            sample: DataSample,
            rng: np.random.Generator | None = None,
    ) -> DataSample:
        stage = AugmentationStage(stage)

        for aug in self.pipeline[stage]:
            sample = aug.apply(sample, rng)

        return sample
