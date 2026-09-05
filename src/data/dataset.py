from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.augmentation.base import AugmentationStage
from src.data.augmentation.pipeline import AugmentationPipeline
from src.data.data_sample import DataSample
from src.data.data_workspace import DataWorkspace
from src.data.preprocess import SamplePreprocessor
from src.data.sample_io import SampleIO
from src.forensic.dct import forensic_maps, luma_qtable


class AIIJCDataset(Dataset):
    """Read, augment, compute DCT maps, and prepare model inputs.

    Test samples additionally contain original_size (int64 [height, width]) and
    image_path (the original CSV value). Default collation yields Bx2 sizes.
    Forensic maps are computed per sample: no cache is justified without profiling,
    and JPEG augmentation changes both pixels and quantization tables each time.
    """

    def __init__(
            self,
            data_workspace: DataWorkspace,
            folded_df: pd.DataFrame,
            train: bool,
            image_size: int,
            seed: int,
            augmentations: AugmentationPipeline | None = None,
            fmap_channels: Sequence[int] | None = None,
            mode: Literal["train", "val", "test"] | None = None,
    ):
        super().__init__()
        self.data_workspace = data_workspace
        self.mode = self._resolve_mode(train, mode)
        self.train = self.mode == "train"
        self.has_targets = self.mode in {"train", "val"}
        self.image_size = image_size
        self.seed = seed
        self.augmentations = augmentations
        self.use_augmentations = self.train and augmentations is not None
        self.fmap_channels = fmap_channels
        self.root = data_workspace.train_root if self.has_targets else data_workspace.test_root
        self.sample_io = SampleIO(self.root, self.has_targets)
        self.preprocessor = SamplePreprocessor(image_size, fmap_channels)
        self.sample_io.validate_dataframe(folded_df)
        self.df = folded_df.reset_index(drop=True)
        self.is_negative = (
            self.df["is_negative"].to_numpy(dtype=bool)
            if "is_negative" in self.df.columns
            else None
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.df.iloc[index]
        image_path = self._image_path(row)
        image = self.load_image(image_path)
        original_size = image.shape[:2]
        sample = DataSample(
            image=image,
            mask=self.load_mask(row, original_size) if self.has_targets else None,
            qtable=luma_qtable(image_path),
        )
        rng = self._make_rng(index)

        # Recompression must precede DCT extraction; maps use the full frame before crop.
        sample = self._augment(AugmentationStage.BEFORE_FORENSICS, sample, rng)
        sample = replace(sample, fmap=forensic_maps(sample.image, sample.qtable))
        sample = self._augment(AugmentationStage.AFTER_FORENSICS, sample, rng)
        sample = self.preprocessor.resize(sample)
        sample = self._augment(AugmentationStage.FINAL, sample, rng)
        output = self.preprocessor.to_output(sample)

        if self.mode == "test":
            output["original_size"] = torch.tensor(original_size, dtype=torch.int64)
            output["image_path"] = str(row["img_path"])
        return output

    def _augment(
            self, stage: AugmentationStage, sample: DataSample, rng: np.random.Generator,
    ) -> DataSample:
        if self.use_augmentations:
            return self.augmentations.apply(stage, sample, rng)
        return sample

    @staticmethod
    def _resolve_mode(train: bool, mode: Literal["train", "val", "test"] | None) -> str:
        if mode is None:
            return "train" if train else "test"
        if mode not in {"train", "val", "test"}:
            raise ValueError(f"unknown dataset mode: {mode}")
        return mode

    # Preserve existing path/loading entry points while keeping IO in one component.
    def _image_path(self, row: pd.Series) -> Path:
        return self.sample_io.image_path(row)

    def _mask_path(self, row: pd.Series) -> Path:
        return self.sample_io.mask_path(row)

    def _row_path(self, row: pd.Series, columns: tuple[str, ...]) -> Path:
        return self.sample_io.row_path(row, columns)

    def load_image(self, path: Path) -> np.ndarray:
        return self.sample_io.load_image(path)

    def load_mask(self, row: pd.Series, image_shape: tuple[int, int]) -> np.ndarray:
        return self.sample_io.load_mask(row, image_shape)

    def _make_rng(self, index: int) -> np.random.Generator:
        worker_seed = torch.initial_seed() % (2 ** 31)
        return np.random.default_rng((self.seed, index, worker_seed))
