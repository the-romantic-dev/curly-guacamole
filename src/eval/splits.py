import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

AREA_BINS = (0.0, 0.01, 0.03, 0.08, 0.20, 1.01)


def make_stratified_val_folds(
    df: pd.DataFrame,
    n_folds: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Добавляет group-aware стратифицированную колонку fold."""
    df = df.loc[~df["broken"]].reset_index(drop=True).copy()
    strata = _make_strata(df, n_folds)

    splitter = StratifiedGroupKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=seed,
    )

    df["fold"] = -1

    for fold, (_, val_idx) in enumerate(
        splitter.split(df, y=strata, groups=df["group_id"])
    ):
        df.loc[val_idx, "fold"] = fold

    check_folds(df, n_folds)
    df["fold"] = df["fold"].astype("int8")
    return df


def train_val_fold_split(
    df: pd.DataFrame,
    fold: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Возвращает train_df и val_df для выбранного fold."""
    if fold not in set(df["fold"].unique()):
        raise ValueError(f"Fold {fold} отсутствует")

    train_df = df.loc[df["fold"] != fold].reset_index(drop=True)
    val_df = df.loc[df["fold"] == fold].reset_index(drop=True)
    return train_df, val_df


def check_folds(df: pd.DataFrame, n_folds: int | None = None) -> None:
    """Проверяет разметку и отсутствие group leakage."""
    if "fold" not in df:
        raise ValueError("Нет колонки fold")

    if (df["fold"] < 0).any():
        raise ValueError("Часть строк не получила fold")

    folds_per_group = df.groupby("group_id")["fold"].nunique()
    leaked = folds_per_group[folds_per_group > 1]

    if not leaked.empty:
        raise ValueError(
            f"{len(leaked)} group_id попали в несколько фолдов: "
            f"{leaked.index[:5].tolist()}"
        )

    if n_folds is not None and df["fold"].nunique() != n_folds:
        raise ValueError(
            f"Ожидалось {n_folds} фолдов, получено {df['fold'].nunique()}"
        )


def summarize_folds(df: pd.DataFrame) -> pd.DataFrame:
    """Основная статистика по каждому fold."""
    return df.groupby("fold").agg(
        samples=("fold", "size"),
        groups=("group_id", "nunique"),
        negative_rate=("is_negative", "mean"),
        median_mask_area=("mask_area", "median"),
    )


def _make_strata(df: pd.DataFrame, n_folds: int) -> pd.Series:
    area_bucket = np.digitize(df["mask_area"], AREA_BINS[1:-1])

    strata = (
        df["domain"].astype(str)
        + "|"
        + df["generator"].astype(str)
        + "|"
        + df["is_negative"].astype(int).astype(str)
        + "|"
        + pd.Series(area_bucket, index=df.index).astype(str)
    )

    counts = strata.value_counts()
    rare = counts[counts < n_folds].index
    return strata.where(~strata.isin(rare), "rare")
