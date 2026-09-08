"""Path resolution and image/mask decoding for competition samples."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def read_image(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """Decode with numpy to support Unicode paths on Windows."""
    try:
        data = np.fromfile(Path(path), dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(data, flags) if data.size else None


class SampleIO:
    def __init__(self, root: Path, has_targets: bool):
        self.root = root
        self.has_targets = has_targets
        self.image_column = "chng_img_path" if has_targets else "img_path"

    def validate_dataframe(self, dataframe: pd.DataFrame) -> None:
        required = [self.image_column, "gt_path"] if self.has_targets else [self.image_column]
        missing = set(required) - set(dataframe.columns)
        if missing:
            raise ValueError(f"dataset is missing required columns: {', '.join(sorted(missing))}")
        for column in required:
            values = dataframe[column]
            invalid = values.isna() | values.astype(str).str.strip().eq("")
            if column == "gt_path" and "target_kind" in dataframe:
                invalid &= dataframe["target_kind"].ne("original_zero")
            if invalid.any():
                raise ValueError(f"dataset column {column!r} contains empty paths")

    def row_path(self, row: pd.Series, columns: tuple[str, ...]) -> Path:
        for column in columns:
            if column in row and not pd.isna(row[column]):
                return self.root / str(row[column]).replace("\\", "/")
        raise KeyError(f"missing path column, expected one of {columns}")

    def image_path(self, row: pd.Series) -> Path:
        return self.row_path(row, (self.image_column,))

    def mask_path(self, row: pd.Series) -> Path:
        if not self.has_targets:
            raise ValueError("test dataset has no mask path")
        return self.row_path(row, ("gt_path",))

    @staticmethod
    def load_image(path: Path) -> np.ndarray:
        image = read_image(path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def load_mask(self, row: pd.Series, image_shape: tuple[int, int], *, strict_size: bool = False) -> np.ndarray:
        if row.get("target_kind") == "original_zero":
            return np.zeros(image_shape, dtype=np.float32)
        path = self.mask_path(row)
        mask = read_image(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Cannot read mask: {path}")
        if mask.shape[:2] != image_shape:
            if strict_size:
                raise ValueError(f"original mask size {mask.shape[:2]} differs from image size {image_shape}: {path}")
            height, width = image_shape
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
        return (mask >= 128).astype(np.float32)
