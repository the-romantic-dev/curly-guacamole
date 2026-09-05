"""Low-level 8x8 DCT operations."""

import numpy as np

from .constants import STRIDE


def _build_dct_matrix() -> np.ndarray:
    """Build orthonormal DCT-II matrix for an 8x8 JPEG block."""
    k = np.arange(STRIDE)

    matrix = np.cos(
        np.pi
        * (2 * k[None, :] + 1)
        * k[:, None]
        / (2 * STRIDE)
    )

    matrix *= np.sqrt(2.0 / STRIDE)
    matrix[0] /= np.sqrt(2.0)

    return matrix.astype(np.float32)


DCT_MATRIX = _build_dct_matrix()


def split_into_blocks(plane: np.ndarray) -> np.ndarray:
    """Convert (H, W) into (H//8, W//8, 8, 8)."""
    blocks_h = plane.shape[0] // STRIDE
    blocks_w = plane.shape[1] // STRIDE

    plane = plane[
        : blocks_h * STRIDE,
        : blocks_w * STRIDE,
    ]

    return (
        plane
        .reshape(blocks_h, STRIDE, blocks_w, STRIDE)
        .transpose(0, 2, 1, 3)
    )


def dct8(blocks: np.ndarray) -> np.ndarray:
    """Apply 2D orthonormal DCT-II to every 8x8 block."""
    return np.einsum(
        "ik,...kl,jl->...ij",
        DCT_MATRIX,
        blocks,
        DCT_MATRIX,
        optimize=True,
    )


def centered_dct(plane: np.ndarray) -> np.ndarray:
    """JPEG-style DCT after subtracting 128 from pixel values."""
    return dct8(split_into_blocks(plane - 128.0))
