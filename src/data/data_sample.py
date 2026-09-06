from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DataSample:
    image: Any
    mask: np.ndarray | None = None
    fmap: np.ndarray | None = None
    qtable: np.ndarray | None = None
    content_size: tuple[int, int] | None = None
