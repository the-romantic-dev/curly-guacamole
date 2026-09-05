"""High-level forensic map API."""

import cv2
import numpy as np

from .constants import CHANNEL_RANGE, STRIDE
from .dct import centered_dct
from .features import build_feature_maps


def forensic_maps(
    rgb: np.ndarray,
    qtable: np.ndarray | None = None,
    *,
    normalize: bool = True,
) -> np.ndarray:
    """Compute 12 DCT forensic maps from an RGB image.

    Args:
        rgb:
            RGB image, shape (H, W, 3).

        qtable:
            JPEG luminance quantization table, shape (8, 8).
            If None, an all-ones table is used.

        normalize:
            Normalize channels using CHANNEL_RANGE and clip to [-4, 4].

    Returns:
        float32 array with shape (12, H//8, W//8).
    """
    _validate_rgb(rgb)

    ycc = _rgb_to_ycrcb_aligned(rgb)
    qtable = _prepare_qtable(qtable)

    dct_y = centered_dct(ycc[..., 0])
    dct_cr = centered_dct(ycc[..., 1])
    dct_cb = centered_dct(ycc[..., 2])

    maps = build_feature_maps(
        dct_y=dct_y,
        dct_cr=dct_cr,
        dct_cb=dct_cb,
        qtable=qtable,
    )

    if normalize:
        maps = _normalize_maps(maps)

    return maps


def _validate_rgb(rgb: np.ndarray) -> None:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(
            f"expected RGB image with shape (H, W, 3), got {rgb.shape}"
        )

    if rgb.shape[0] < STRIDE or rgb.shape[1] < STRIDE:
        raise ValueError(
            f"image must be at least {STRIDE}x{STRIDE}, got {rgb.shape}"
        )


def _rgb_to_ycrcb_aligned(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB to float32 YCrCb and align spatial size to 8."""
    ycc = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2YCrCb,
    ).astype(np.float32)

    height = (ycc.shape[0] // STRIDE) * STRIDE
    width = (ycc.shape[1] // STRIDE) * STRIDE

    return ycc[:height, :width]


def _prepare_qtable(qtable: np.ndarray | None) -> np.ndarray:
    """Convert qtable to float32 shape (8, 8)."""
    if qtable is None:
        return np.ones((8, 8), dtype=np.float32)

    qtable = np.asarray(qtable, dtype=np.float32)

    if qtable.shape != (8, 8):
        raise ValueError(
            f"qtable must have shape (8, 8), got {qtable.shape}"
        )

    return qtable


def _normalize_maps(maps: np.ndarray) -> np.ndarray:
    maps = maps / CHANNEL_RANGE[:, None, None]
    return np.clip(maps, -4.0, 4.0).astype(np.float32)
