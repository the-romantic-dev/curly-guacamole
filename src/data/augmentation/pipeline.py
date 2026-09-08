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
            *, total_epochs: int | None = None,
            jpeg_qtable_order: str = "legacy_zigzag",
    ):
        self.config = (
            config
            if isinstance(config, AugmentationConfig)
            else AugmentationConfig(**dict(config))
        )
        self.total_epochs = total_epochs
        self.jpeg_qtable_order = jpeg_qtable_order
        self.epoch = 0
        if self.config.final_full_frame_epochs and total_epochs is None:
            raise ValueError("total_epochs is required for the final full-frame phase")
        self.full_frame_flip = RandomRotateFlip(full_frame=True)
        self.pipeline: dict[AugmentationStage, list[AIIJCAugmentation]] = {
            AugmentationStage.BEFORE_FORENSICS: [
                RandomJPEGRecompression(
                    quality_range=self.config.jpeg_recompression_quality_range,
                    probability=self.config.jpeg_recompression_probability,
                    jpeg_qtable_order=jpeg_qtable_order,
                )
            ],
            AugmentationStage.AFTER_FORENSICS: [
                RandomDCTAlignedCrop(crop_scale_range=self.config.crop_scale_range,
                                     full_frame=self.config.full_frame,
                                     foreground_probability=self.config.foreground_crop_probability),
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
        if stage == AugmentationStage.AFTER_FORENSICS:
            if rng is None:
                raise ValueError("after_forensics augmentations require rng")
            if rng.random() < self.full_frame_probability:
                return self.full_frame_flip.apply(sample, rng)

        for aug in self.pipeline[stage]:
            sample = aug.apply(sample, rng)

        return sample

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    @property
    def full_frame_probability(self) -> float:
        final_phase = (self.total_epochs is not None
                       and self.config.final_full_frame_epochs > 0
                       and self.epoch >= self.total_epochs - self.config.final_full_frame_epochs)
        if self.config.full_frame or final_phase:
            return 1.0
        return self.config.full_frame_probability
