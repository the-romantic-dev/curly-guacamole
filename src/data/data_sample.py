from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from src.forensic.jpeg_input import JPEGInput


@dataclass(frozen=True)
class DataSample:
    image: Any
    mask: np.ndarray | None = None
    fmap: np.ndarray | None = None
    qtable: np.ndarray | None = None
    content_size: tuple[int, int] | None = None
    jpeg: JPEGInput | None = None
