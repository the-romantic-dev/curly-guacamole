from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.data.augmentation.base import AugmentationConfig
from src.training.metric import DEFAULT_AREA_GRID, DEFAULT_CLS_GRID, DEFAULT_MASK_GRID


@dataclass(frozen=True)
class PathsConfig:
    data_path: Path
    runs_path: Path
    run_name: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, base_dir: Path) -> PathsConfig:
        data = _mapping(data, "paths")
        _check_keys(data, {"data_path", "runs_path", "run_name"}, "paths")
        return cls(
            data_path=_path(_required(data, "data_path", "paths"), base_dir),
            runs_path=_path(_required(data, "runs_path", "paths"), base_dir),
            run_name=_non_empty_str(_required(data, "run_name", "paths"), "paths.run_name"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "data_path": str(self.data_path),
            "runs_path": str(self.runs_path),
            "run_name": self.run_name,
        }


@dataclass(frozen=True)
class ModelConfig:
    encoder_name: str
    decoder_channels: tuple[int, ...]
    forensic_channels: tuple[int, ...]
    aux_weight: float
    norm: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ModelConfig:
        data = _mapping(data, "model")
        _check_keys(
            data,
            {"encoder_name", "decoder_channels", "forensic_channels", "aux_weight", "norm"},
            "model",
        )
        config = cls(
            encoder_name=_non_empty_str(_required(data, "encoder_name", "model"), "model.encoder_name"),
            decoder_channels=_int_tuple(_required(data, "decoder_channels", "model"), "model.decoder_channels"),
            forensic_channels=_int_tuple(_required(data, "forensic_channels", "model"), "model.forensic_channels"),
            aux_weight=float(_required(data, "aux_weight", "model")),
            norm=_non_empty_str(_required(data, "norm", "model"), "model.norm"),
        )
        if not config.decoder_channels:
            raise ValueError("model.decoder_channels must not be empty")
        if len(config.forensic_channels) != 3:
            raise ValueError("model.forensic_channels must contain exactly 3 values")
        if config.aux_weight < 0:
            raise ValueError("model.aux_weight must be non-negative")
        if config.norm not in {"batch", "group"}:
            raise ValueError("model.norm must be 'batch' or 'group'")
        return config

    def to_dict(self) -> dict[str, Any]:
        return {
            "encoder_name": self.encoder_name,
            "decoder_channels": self.decoder_channels,
            "forensic_channels": self.forensic_channels,
            "aux_weight": self.aux_weight,
            "norm": self.norm,
        }


@dataclass(frozen=True)
class DatasetConfig:
    fold: int
    n_folds: int
    image_size: int

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DatasetConfig:
        data = _mapping(data, "dataset")
        _check_keys(data, {"fold", "n_folds", "image_size"}, "dataset")
        config = cls(
            fold=int(_required(data, "fold", "dataset")),
            n_folds=int(_required(data, "n_folds", "dataset")),
            image_size=int(_required(data, "image_size", "dataset")),
        )
        if config.n_folds < 2:
            raise ValueError("dataset.n_folds must be at least 2")
        if not 0 <= config.fold < config.n_folds:
            raise ValueError("dataset.fold must be in [0, n_folds)")
        if config.image_size < 8 or config.image_size % 8 != 0:
            raise ValueError("dataset.image_size must be divisible by 8 and at least 8")
        return config

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold": self.fold,
            "n_folds": self.n_folds,
            "image_size": self.image_size,
        }


@dataclass(frozen=True)
class TrainConfig:
    device: str
    workers: int
    encoder_lr: float
    fmap_lr: float
    lr: float
    weight_decay: float
    epochs: int
    epoch_size: int
    val_frac: float
    val_keep_negatives: bool
    negative_fraction: float
    batch_size: int
    accum_steps: int
    warmup_frac: float
    min_lr_factor: float
    amp: str
    ema_decay: float
    grad_clip: float
    resume: bool

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TrainConfig:
        data = _mapping(data, "train")
        _check_keys(
            data,
            {
                "device",
                "workers",
                "encoder_lr",
                "fmap_lr",
                "lr",
                "weight_decay",
                "epochs",
                "epoch_size",
                "val_frac",
                "val_keep_negatives",
                "negative_fraction",
                "batch_size",
                "accum_steps",
                "warmup_frac",
                "min_lr_factor",
                "amp",
                "ema_decay",
                "grad_clip",
                "resume",
            },
            "train",
        )
        config = cls(
            device=_non_empty_str(_required(data, "device", "train"), "train.device"),
            workers=int(_required(data, "workers", "train")),
            encoder_lr=float(_required(data, "encoder_lr", "train")),
            fmap_lr=float(_required(data, "fmap_lr", "train")),
            lr=float(_required(data, "lr", "train")),
            weight_decay=float(_required(data, "weight_decay", "train")),
            epochs=int(_required(data, "epochs", "train")),
            epoch_size=int(_required(data, "epoch_size", "train")),
            val_frac=float(_required(data, "val_frac", "train")),
            val_keep_negatives=bool(_required(data, "val_keep_negatives", "train")),
            negative_fraction=float(_required(data, "negative_fraction", "train")),
            batch_size=int(_required(data, "batch_size", "train")),
            accum_steps=int(_required(data, "accum_steps", "train")),
            warmup_frac=float(_required(data, "warmup_frac", "train")),
            min_lr_factor=float(_required(data, "min_lr_factor", "train")),
            amp=_non_empty_str(_required(data, "amp", "train"), "train.amp"),
            ema_decay=float(_required(data, "ema_decay", "train")),
            grad_clip=float(_required(data, "grad_clip", "train")),
            resume=bool(_required(data, "resume", "train")),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not (self.device == "cpu" or self.device == "mps" or self.device.startswith("cuda")):
            raise ValueError("train.device must be 'cpu', 'mps', 'cuda' or 'cuda:<index>'")
        if self.workers < 0:
            raise ValueError("train.workers must be non-negative")
        if min(self.encoder_lr, self.fmap_lr, self.lr) <= 0:
            raise ValueError("learning rates must be positive")
        if self.weight_decay < 0:
            raise ValueError("train.weight_decay must be non-negative")
        if self.epochs <= 0:
            raise ValueError("train.epochs must be positive")
        if self.epoch_size <= 0:
            raise ValueError("train.epoch_size must be positive")
        if not 0.0 <= self.val_frac <= 1.0:
            raise ValueError("train.val_frac must be in [0, 1]")
        if not 0.0 < self.negative_fraction < 1.0:
            raise ValueError("train.negative_fraction must be in (0, 1)")
        if self.batch_size <= 0:
            raise ValueError("train.batch_size must be positive")
        if self.accum_steps <= 0:
            raise ValueError("train.accum_steps must be positive")
        if not 0.0 <= self.warmup_frac <= 1.0:
            raise ValueError("train.warmup_frac must be in [0, 1]")
        if not 0.0 <= self.min_lr_factor <= 1.0:
            raise ValueError("train.min_lr_factor must be in [0, 1]")
        if self.amp not in {"off", "fp16", "bf16"}:
            raise ValueError("train.amp must be 'off', 'fp16' or 'bf16'")
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError("train.ema_decay must be in [0, 1)")
        if self.grad_clip < 0:
            raise ValueError("train.grad_clip must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "workers": self.workers,
            "encoder_lr": self.encoder_lr,
            "fmap_lr": self.fmap_lr,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "epochs": self.epochs,
            "epoch_size": self.epoch_size,
            "val_frac": self.val_frac,
            "val_keep_negatives": self.val_keep_negatives,
            "negative_fraction": self.negative_fraction,
            "batch_size": self.batch_size,
            "accum_steps": self.accum_steps,
            "warmup_frac": self.warmup_frac,
            "min_lr_factor": self.min_lr_factor,
            "amp": self.amp,
            "ema_decay": self.ema_decay,
            "grad_clip": self.grad_clip,
            "resume": self.resume,
        }


@dataclass(frozen=True)
class EvalConfig:
    n_bins: int = 256
    mask_thresholds: tuple[float, ...] = DEFAULT_MASK_GRID
    cls_thresholds: tuple[float, ...] = DEFAULT_CLS_GRID
    min_areas: tuple[float, ...] = DEFAULT_AREA_GRID

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> EvalConfig:
        defaults = cls()
        if data is None:
            return defaults
        data = _mapping(data, "eval")
        _check_keys(data, {"n_bins", "mask_thresholds", "cls_thresholds", "min_areas"}, "eval")
        config = cls(
            n_bins=int(data.get("n_bins", defaults.n_bins)),
            mask_thresholds=_float_tuple(data.get("mask_thresholds", defaults.mask_thresholds), "eval.mask_thresholds"),
            cls_thresholds=_float_tuple(data.get("cls_thresholds", defaults.cls_thresholds), "eval.cls_thresholds"),
            min_areas=_float_tuple(data.get("min_areas", defaults.min_areas), "eval.min_areas"),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.n_bins <= 0:
            raise ValueError("eval.n_bins must be positive")
        _check_probability_grid(self.mask_thresholds, "eval.mask_thresholds")
        _check_probability_grid(self.cls_thresholds, "eval.cls_thresholds")
        _check_probability_grid(self.min_areas, "eval.min_areas")

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_bins": self.n_bins,
            "mask_thresholds": self.mask_thresholds,
            "cls_thresholds": self.cls_thresholds,
            "min_areas": self.min_areas,
        }


@dataclass(frozen=True)
class ExperimentConfig:
    paths: PathsConfig
    seed: int
    model: ModelConfig
    augmentation: AugmentationConfig
    dataset: DatasetConfig
    train: TrainConfig
    eval: EvalConfig

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, base_dir: str | Path = ".") -> ExperimentConfig:
        data = _mapping(data, "experiment")
        _check_keys(data, {"paths", "seed", "model", "augmentation", "dataset", "train", "eval"}, "experiment")
        base_dir = Path(base_dir)
        config = cls(
            paths=PathsConfig.from_dict(_required(data, "paths", "experiment"), base_dir=base_dir),
            seed=int(_required(data, "seed", "experiment")),
            model=ModelConfig.from_dict(_required(data, "model", "experiment")),
            augmentation=_augmentation_from_dict(_required(data, "augmentation", "experiment")),
            dataset=DatasetConfig.from_dict(_required(data, "dataset", "experiment")),
            train=TrainConfig.from_dict(_required(data, "train", "experiment")),
            eval=EvalConfig.from_dict(data.get("eval")),
        )
        if config.seed < 0:
            raise ValueError("seed must be non-negative")
        return config

    def to_dict(self) -> dict[str, Any]:
        return {
            "paths": self.paths.to_dict(),
            "seed": self.seed,
            "model": self.model.to_dict(),
            "augmentation": _augmentation_to_dict(self.augmentation),
            "dataset": self.dataset.to_dict(),
            "train": self.train.to_dict(),
            "eval": self.eval.to_dict(),
        }

    def to_flat_dict(self) -> dict[str, Any]:
        plain = {
            "data_path": str(self.paths.data_path),
            "runs_path": str(self.paths.runs_path),
            "run_name": self.paths.run_name,
            "seed": self.seed,
        }
        for section in (self.model, self.augmentation, self.dataset, self.train, self.eval):
            if isinstance(section, AugmentationConfig):
                plain.update(_augmentation_to_dict(section))
            else:
                plain.update(section.to_dict())
        return plain


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return ExperimentConfig.from_dict(data, base_dir=path.parent)


def _augmentation_from_dict(data: Mapping[str, Any]) -> AugmentationConfig:
    data = _mapping(data, "augmentation")
    _check_keys(
        data,
        {
            "crop_scale_range",
            "jpeg_recompression_probability",
            "jpeg_recompression_quality_range",
            "full_frame",
        },
        "augmentation",
    )
    config = AugmentationConfig(
        crop_scale_range=_float_tuple(
            _required(data, "crop_scale_range", "augmentation"),
            "augmentation.crop_scale_range",
        ),
        jpeg_recompression_probability=float(
            _required(data, "jpeg_recompression_probability", "augmentation")
        ),
        jpeg_recompression_quality_range=_int_tuple(
            _required(data, "jpeg_recompression_quality_range", "augmentation"),
            "augmentation.jpeg_recompression_quality_range",
        ),
        full_frame=bool(_required(data, "full_frame", "augmentation")),
    )
    if len(config.crop_scale_range) != 2:
        raise ValueError("augmentation.crop_scale_range must contain exactly 2 values")
    crop_min, crop_max = config.crop_scale_range
    if not 0.0 < crop_min <= crop_max <= 1.0:
        raise ValueError("augmentation.crop_scale_range must satisfy 0 < min <= max <= 1")
    if not 0.0 <= config.jpeg_recompression_probability <= 1.0:
        raise ValueError("augmentation.jpeg_recompression_probability must be in [0, 1]")
    if len(config.jpeg_recompression_quality_range) != 2:
        raise ValueError("augmentation.jpeg_recompression_quality_range must contain exactly 2 values")
    quality_min, quality_max = config.jpeg_recompression_quality_range
    if not 1 <= quality_min <= quality_max <= 100:
        raise ValueError("augmentation.jpeg_recompression_quality_range must satisfy 1 <= min <= max <= 100")
    return config


def _augmentation_to_dict(config: AugmentationConfig) -> dict[str, Any]:
    return {
        "crop_scale_range": config.crop_scale_range,
        "jpeg_recompression_probability": config.jpeg_recompression_probability,
        "jpeg_recompression_quality_range": config.jpeg_recompression_quality_range,
        "full_frame": config.full_frame,
    }


def _mapping(data: Any, section: str) -> Mapping[str, Any]:
    if not isinstance(data, Mapping):
        raise TypeError(f"{section} must be a mapping")
    return data


def _check_keys(data: Mapping[str, Any], allowed: set[str], section: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"unknown keys in {section}: {', '.join(unknown)}")
    missing = sorted(allowed - set(data))
    if missing:
        raise ValueError(f"missing keys in {section}: {', '.join(missing)}")


def _required(data: Mapping[str, Any], key: str, section: str) -> Any:
    if key not in data:
        raise ValueError(f"missing key in {section}: {key}")
    return data[key]


def _non_empty_str(value: Any, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must not be empty")
    return text


def _path(value: Any, base_dir: Path) -> Path:
    path = Path(os.path.expandvars(str(value))).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _int_tuple(value: Any, name: str) -> tuple[int, ...]:
    return tuple(int(item) for item in _sequence(value, name))


def _float_tuple(value: Any, name: str) -> tuple[float, ...]:
    return tuple(float(item) for item in _sequence(value, name))


def _sequence(value: Any, name: str) -> tuple[Any, ...]:
    if isinstance(value, str) or not hasattr(value, "__iter__"):
        raise TypeError(f"{name} must be a sequence")
    result = tuple(value)
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def _check_probability_grid(values: tuple[float, ...], name: str) -> None:
    if any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError(f"{name} values must be in [0, 1]")


__all__ = [
    "AugmentationConfig",
    "DatasetConfig",
    "EvalConfig",
    "ExperimentConfig",
    "ModelConfig",
    "PathsConfig",
    "TrainConfig",
    "load_experiment_config",
]

