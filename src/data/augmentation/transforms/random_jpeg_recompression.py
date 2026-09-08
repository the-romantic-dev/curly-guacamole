from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np

from src.data.augmentation.base import AIIJCAugmentation, AugmentationStage, require_rng
from src.data.data_sample import DataSample
from src.forensic.dct import luma_qtable


class RandomJPEGRecompression(AIIJCAugmentation):
    def __init__(self, quality_range: tuple[int, int], probability: float,
                 *, jpeg_qtable_order: str = "legacy_zigzag") -> None:
        super().__init__()
        self.quality_range = quality_range
        self.probability = probability
        self.jpeg_qtable_order = jpeg_qtable_order

    def jpeg_recompression(
            self,
            image: np.ndarray,
            rng: np.random.Generator | None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """JPEG re-encode image and extract the resulting luminance qtable."""
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        quality = int(rng.integers(*self.quality_range))
        ok, buffer = cv2.imencode(
            ".jpg",
            bgr,
            [cv2.IMWRITE_JPEG_QUALITY, quality],
        )
        if not ok:
            return image, None

        jpeg_bytes = buffer.tobytes()
        decoded = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
        decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)

        return decoded, luma_qtable(jpeg_bytes, order=self.jpeg_qtable_order)

    def apply(self, sample: DataSample, rng: np.random.Generator | None = None) -> DataSample:
        if self.probability <= 0:
            return sample

        rng = require_rng(rng, AugmentationStage.BEFORE_FORENSICS)
        if rng.random() >= self.probability:
            return sample

        image, qtable = self.jpeg_recompression(sample.image, rng)
        return replace(sample, image=image, qtable=qtable)
