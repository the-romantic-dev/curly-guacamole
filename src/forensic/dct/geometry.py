"""Spatial operations for forensic maps."""

import cv2
import numpy as np

from .constants import STRIDE


def crop_fmaps(
    maps: np.ndarray,
    top: int,
    left: int,
    height: int,
    width: int,
) -> np.ndarray:
    """Crop forensic maps using coordinates from the original image.

    All coordinates and sizes must be divisible by 8.
    """
    values = {
        "top": top,
        "left": left,
        "height": height,
        "width": width,
    }

    for name, value in values.items():
        if value % STRIDE != 0:
            raise ValueError(
                f"{name}={value} must be divisible by {STRIDE}"
            )

    return maps[
        :,
        top // STRIDE : (top + height) // STRIDE,
        left // STRIDE : (left + width) // STRIDE,
    ]


def resize_fmaps(
    maps: np.ndarray,
    image_size: int,
) -> np.ndarray:
    """Resize forensic maps to the 1/8 grid of a square model input."""
    target_size = max(1, image_size // STRIDE)

    if maps.shape[1:] == (target_size, target_size):
        return maps

    maps_hwc = np.ascontiguousarray(
        maps.transpose(1, 2, 0)
    )

    resized = cv2.resize(
        maps_hwc,
        (target_size, target_size),
        interpolation=cv2.INTER_LINEAR,
    )

    if resized.ndim == 2:
        resized = resized[..., None]

    return np.ascontiguousarray(
        resized.transpose(2, 0, 1)
    )
