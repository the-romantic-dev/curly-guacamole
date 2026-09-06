from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.config import ExperimentConfig, ModelConfig, TrainConfig
from src.data.augmentation.pipeline import AugmentationPipeline
from src.data.collation import ValidationCollator
from src.data.data_workspace import DataWorkspace
from src.data.dataset import AIIJCDataset

if TYPE_CHECKING:
    from src.modules.segmenter import Segmenter


class DataLoaderThreadLimits:
    """Keep CPU libraries single-threaded in the parent and loader workers."""

    @staticmethod
    def apply(worker_id: int | None = None) -> None:
        for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[variable] = "1"
        cv2.setNumThreads(1)
        torch.set_num_threads(1)
        # Also limit libraries already imported by a notebook or forked worker.
        threadpool_limits(limits=1)


@dataclass(frozen=True)
class AmpContext:
    device: torch.device
    dtype: torch.dtype
    enabled: bool
    scaler_enabled: bool

    def autocast(self) -> Any:
        return torch.amp.autocast(
            self.device.type,
            dtype=self.dtype,
            enabled=self.enabled,
        )

    def scaler(self) -> torch.amp.GradScaler:
        return torch.amp.GradScaler(
            self.device.type,
            enabled=self.scaler_enabled,
        )


def build_model(config: ModelConfig, *, pretrained: bool = True) -> Segmenter:
    from src.modules.segmenter import Segmenter

    return Segmenter(
        encoder_name=config.encoder_name,
        decoder_channels=config.decoder_channels,
        forensic_channels=config.forensic_channels,
        norm=config.norm,
        aux_weight=config.aux_weight,
        use_forensics=config.use_forensics,
        dct_aux_weight=config.dct_aux_weight,
        pretrained=pretrained,
        decoder_name=config.decoder_name,
        decoder_embed_dim=config.decoder_embed_dim,
        decoder_kwargs=config.decoder_kwargs,
    )


def build_datasets(
    config: ExperimentConfig,
    data_workspace: DataWorkspace,
    train_df,
    val_df,
) -> tuple[AIIJCDataset, AIIJCDataset]:
    augmentations = AugmentationPipeline(config.augmentation, total_epochs=config.train.epochs)

    train_ds = AIIJCDataset(
        data_workspace=data_workspace,
        folded_df=train_df,
        train=True,
        image_size=config.dataset.image_size,
        seed=config.seed,
        augmentations=augmentations,
        fmap_channels=None,
        use_forensics=config.model.use_forensics,
        mode="train",
        resize_mode=config.dataset.resize_mode,
    )

    val_ds = AIIJCDataset(
        data_workspace=data_workspace,
        folded_df=val_df,
        train=False,
        image_size=config.dataset.image_size,
        seed=config.seed,
        augmentations=None,
        fmap_channels=None,
        use_forensics=config.model.use_forensics,
        mode="val",
        resize_mode=config.dataset.resize_mode,
        original_targets=config.eval.resolution == "original",
    )

    return train_ds, val_ds


def build_sampler(
    config: TrainConfig,
    train_ds: AIIJCDataset,
) -> WeightedRandomSampler:
    neg = train_ds.is_negative
    if neg is None:
        raise ValueError("train dataset must expose is_negative for weighted sampling")

    n_neg = int(neg.sum())
    n_pos = int((~neg).sum())
    if n_neg == 0 or n_pos == 0:
        raise ValueError(f"weighted sampling requires both classes, got pos={n_pos}, neg={n_neg}")

    neg_fr = config.negative_fraction
    weights = np.where(neg, neg_fr / n_neg, (1 - neg_fr) / n_pos)
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=config.epoch_size,
        replacement=True,
    )


def build_loaders(
    config: TrainConfig,
    train_ds: AIIJCDataset,
    val_ds: AIIJCDataset,
) -> tuple[DataLoader, DataLoader]:
    DataLoaderThreadLimits.apply()
    pin_memory = torch.device(config.device).type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        sampler=build_sampler(config, train_ds),
        drop_last=True,
        num_workers=config.workers,
        pin_memory=pin_memory,
        persistent_workers=config.workers > 0,
        worker_init_fn=DataLoaderThreadLimits.apply,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size * 2,
        shuffle=False,
        num_workers=config.workers,
        pin_memory=pin_memory,
        persistent_workers=config.workers > 0,
        worker_init_fn=DataLoaderThreadLimits.apply,
        collate_fn=ValidationCollator(),
    )
    return train_loader, val_loader


def build_optimizer(
    config: TrainConfig,
    model,
) -> torch.optim.AdamW:
    """AdamW with separate LR/weight decay for encoder, forensic branch and decoder."""
    groups = {"enc": [], "enc_nd": [], "fmap": [], "fmap_nd": [], "dec": [], "dec_nd": []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        no_decay = parameter.ndim <= 1 or name.endswith(("channel_gate", "gamma"))
        if name.startswith(("forensic_fusion.", "branch.", "fuse.")):
            groups["fmap_nd" if no_decay else "fmap"].append(parameter)
        elif name.startswith("encoder."):
            groups["enc_nd" if no_decay else "enc"].append(parameter)
        else:
            groups["dec_nd" if no_decay else "dec"].append(parameter)

    param_groups = [
        {"params": groups["enc"], "lr": config.encoder_lr, "weight_decay": config.weight_decay},
        {"params": groups["enc_nd"], "lr": config.encoder_lr, "weight_decay": 0.0},
        {"params": groups["fmap"], "lr": config.fmap_lr, "weight_decay": config.weight_decay},
        {"params": groups["fmap_nd"], "lr": config.fmap_lr, "weight_decay": 0.0},
        {"params": groups["dec"], "lr": config.lr, "weight_decay": config.weight_decay},
        {"params": groups["dec_nd"], "lr": config.lr, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW([group for group in param_groups if group["params"]])


def build_scheduler(
    config: TrainConfig,
    optimizer: torch.optim.Optimizer,
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    steps_per_epoch = max(1, int(steps_per_epoch))
    total_steps = max(1, steps_per_epoch * config.epochs)
    warmup = int(config.warmup_frac * total_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return config.min_lr_factor + (1 - config.min_lr_factor) * 0.5 * (
            1 + math.cos(math.pi * min(progress, 1.0))
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_amp(
    config: TrainConfig,
    device: torch.device | None = None,
) -> AmpContext:
    device = torch.device(config.device) if device is None else device
    return AmpContext(
        device=device,
        dtype=torch.bfloat16 if config.amp == "bf16" else torch.float16,
        enabled=device.type == "cuda" and config.amp != "off",
        scaler_enabled=device.type == "cuda" and config.amp == "fp16",
    )


def build_ema(
    config: TrainConfig,
    model,
) -> torch.optim.swa_utils.AveragedModel:
    decay = config.ema_decay
    return torch.optim.swa_utils.AveragedModel(
        model,
        avg_fn=lambda avg, cur, _: decay * avg + (1 - decay) * cur,
    )


__all__ = [
    "AmpContext",
    "build_amp",
    "build_datasets",
    "build_ema",
    "build_loaders",
    "build_model",
    "build_optimizer",
    "build_sampler",
    "build_scheduler",
]
