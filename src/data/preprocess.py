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
from src.data.letterbox import Letterbox
from src.data.targets import mask_to_tensor
from src.forensic.dct import resize_fmaps


def image_resize(sample: DataSample, size: int, *, rgb_size: int | None = None) -> DataSample:
    # size = require_model_size(sample)
    image = sample.image
    fmap = sample.fmap

    rgb_size = size if rgb_size is None else rgb_size
    image = cv2.resize(image, (rgb_size, rgb_size), interpolation=cv2.INTER_LINEAR)
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

    def __init__(self, image_size: int, fmap_channels: Sequence[int] | None = None,
                 resize_mode: str = "stretch", *, strided_resize: bool = False):
        self.image_size = image_size
        self.strided_resize = strided_resize
        if strided_resize and resize_mode != 'stretch':
            raise ValueError('strided_resize requires stretch geometry')
        self.fmap_channels = fmap_channels
        if resize_mode not in {"stretch", "letterbox"}:
            raise ValueError("resize_mode must be 'stretch' or 'letterbox'")
        self.letterbox = Letterbox(image_size) if resize_mode == "letterbox" else None

    def resize(self, sample: DataSample) -> DataSample:
        if self.letterbox is not None:
            return self.letterbox.apply(sample)
        return image_resize(sample, self.image_size,
                            rgb_size=self.image_size * 2 if self.strided_resize else None)

    def to_output(self, sample: DataSample) -> dict[str, torch.Tensor]:
        sample = (replace(sample, image=image_to_tensor(sample.image))
                  if self.strided_resize else imagenet_normalize(sample))
        fmap = require_fmap(sample)
        if self.fmap_channels is not None:
            fmap = fmap[list(self.fmap_channels)]
        output = {"image": sample.image, "fmap": torch.from_numpy(np.ascontiguousarray(fmap))}
        if sample.content_size is not None:
            h, w = sample.content_size
            valid = torch.zeros((1, self.image_size, self.image_size), dtype=torch.bool)
            valid[:, :h, :w] = True
            # Reset padding after photometric augmentation and normalization.
            output["image"] = output["image"].masked_fill(~valid, 0)
            output["valid_mask"] = valid
            output["content_size"] = torch.tensor([h, w], dtype=torch.int64)
        if sample.mask is not None:
            mask = mask_to_tensor(sample.mask)
            output["mask"] = mask
            output["label"] = torch.tensor([float(mask.max() > 0)], dtype=torch.float32)
        return output

