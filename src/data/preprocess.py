"""Resize and convert samples while preserving the baseline interpolation contract."""

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from src.data.augmentation.base import require_fmap
from src.data.data_sample import DataSample
from src.data.targets import mask_to_tensor
from src.forensic.dct import resize_fmaps


def image_resize(sample: DataSample, size: int) -> DataSample:
    # size = require_model_size(sample)
    image = sample.image
    fmap = sample.fmap

    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    fmap = resize_fmaps(fmap, size)
    mask = (
        cv2.resize(sample.mask, (size, size), interpolation=cv2.INTER_LINEAR)
        if sample.mask is not None
        else None
    )

    return replace(sample, image=image, mask=mask, fmap=fmap)


def imagenet_normalize(sample: DataSample) -> DataSample:
    transform = A.Compose([
        A.Normalize(
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        ),
        ToTensorV2(),
    ])

    image = transform(image=sample.image)["image"]
    return replace(sample, image=image)


def image_to_tensor(image: Any):
    if not isinstance(image, np.ndarray):
        return image

    return torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0


class SamplePreprocessor:
    """Prepare synchronized samples for the model and optional target heads."""

    def __init__(self, image_size: int, fmap_channels: Sequence[int] | None = None):
        self.image_size = image_size
        self.fmap_channels = fmap_channels

    def resize(self, sample: DataSample) -> DataSample:
        return image_resize(sample, self.image_size)

    def to_output(self, sample: DataSample) -> dict[str, torch.Tensor]:
        sample = imagenet_normalize(sample)
        fmap = require_fmap(sample)
        if self.fmap_channels is not None:
            fmap = fmap[list(self.fmap_channels)]
        output = {"image": sample.image, "fmap": torch.from_numpy(np.ascontiguousarray(fmap))}
        if sample.mask is not None:
            mask = mask_to_tensor(sample.mask)
            output["mask"] = mask
            output["label"] = torch.tensor([float(mask.max() > 0)], dtype=torch.float32)
        return output

