from pathlib import Path

import pandas as pd


class DataWorkspace:
    def __init__(self, data_path: Path):
        self.data_path = data_path

    @property
    def train_root(self) -> Path:
        return self.data_path / "train_stage1"

    @property
    def test_root(self) -> Path:
        return self.data_path / "test_stage1" / "test_stage1"

    @property
    def train_csv(self) -> pd.DataFrame:
        path = self.train_root / "stage1" / "train.csv"
        return pd.read_csv(path)

    @property
    def test_csv(self) -> pd.DataFrame:
        path = self.test_root / "test.csv"
        return pd.read_csv(path)
    