"""Aspect-preserving resize with bottom/right padding and its inverse."""

from dataclasses import replace

import cv2
import numpy as np
import torch.nn.functional as F

from src.data.data_sample import DataSample


class Letterbox:
    def __init__(self, size: int):
        self.size = size

    def apply(self, sample: DataSample) -> DataSample:
        height, width = sample.image.shape[:2]
        scale = self.size / max(height, width)
        h = min(self.size, max(1, round(height * scale)))
        w = min(self.size, max(1, round(width * scale)))
        image = np.zeros((self.size, self.size, 3), dtype=sample.image.dtype)
        image[:h, :w] = cv2.resize(sample.image, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = None
        if sample.mask is not None:
            mask = np.zeros((self.size, self.size), dtype=sample.mask.dtype)
            mask[:h, :w] = cv2.resize(sample.mask, (w, h), interpolation=cv2.INTER_LINEAR)
        if sample.fmap is None:
            raise ValueError("letterbox requires forensic maps")

        # Sample the original block grid directly, without allocating full-resolution
        # 12-channel maps. The origin stays 8-aligned; partial edge blocks get coverage weights.
        grid = np.arange(self.size // 8, dtype=np.float32) + 0.5
        map_x, map_y = np.meshgrid(grid * width / w - 0.5, grid * height / h - 0.5)
        fmap = cv2.remap(sample.fmap.transpose(1, 2, 0), map_x, map_y,
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        coverage_y = np.clip(h / 8 - np.arange(self.size // 8), 0, 1)
        coverage_x = np.clip(w / 8 - np.arange(self.size // 8), 0, 1)
        fmap *= (coverage_y[:, None] * coverage_x[None, :])[..., None]
        return replace(sample, image=image, mask=mask,
                       fmap=np.ascontiguousarray(fmap.transpose(2, 0, 1)), content_size=(h, w))

    @staticmethod
    def crop(probability, content_size=None):
        if content_size is None:
            return probability
        h, w = (int(value) for value in content_size)
        if not (0 < h <= probability.shape[-2] and 0 < w <= probability.shape[-1]):
            raise ValueError("content_size must fit within the prediction")
        return probability[..., :h, :w]

    @staticmethod
    def restore(probability, size, content_size=None):
        return F.interpolate(Letterbox.crop(probability, content_size), size=size,
                             mode="bilinear", align_corners=False)
