"""JPEG metadata helpers."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image

from .constants import ZIGZAG


def luma_qtable(
    source: str | Path | bytes | bytearray | memoryview,
    *, order: str = "legacy_zigzag",
) -> np.ndarray | None:
    """Read JPEG luminance quantization table as an 8x8 array.

    Pillow already returns natural frequency order. The legacy permutation is
    retained for checkpoints trained with the old feature protocol.
    Returns None for non-JPEG inputs or when the table cannot be read.
    """
    if order not in {"natural", "legacy_zigzag"}:
        raise ValueError("JPEG qtable order must be 'natural' or 'legacy_zigzag'")
    if isinstance(source, (bytes, bytearray, memoryview)):
        source = io.BytesIO(bytes(source))

    try:
        with Image.open(source) as image:
            tables = getattr(image, "quantization", None)

        if not tables:
            return None

        table = np.asarray(tables[0], dtype=np.float32)
        return table.reshape(8, 8) if order == "natural" else table[ZIGZAG]

    except Exception:
        return None
