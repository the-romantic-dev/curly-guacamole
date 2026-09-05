from pathlib import Path

import pytest

from src.config import ExperimentConfig, load_experiment_config


def test_load_baseline_config():
    config = load_experiment_config("configs/baseline.yaml")

    assert config.paths.data_path == Path("D:/Challenges/AIIJC2026/data")
    assert config.paths.runs_path == Path("runs").resolve()
    assert config.paths.run_name == "pvt_v2_b2_full_frame"
    assert config.seed == 42
    assert config.model.encoder_name == "pvt_v2_b2"
    assert config.model.decoder_channels == (128, 64, 32, 16, 16)
    assert config.model.forensic_channels == (64, 96, 128)
    assert config.augmentation.crop_scale_range == (0.35, 1.0)
    assert config.dataset.image_size == 640
    assert config.train.amp == "bf16"
    assert config.eval.n_bins == 256
    assert config.eval.mask_thresholds[0] == 0.05
    assert config.eval.mask_thresholds[-1] == 0.95
    assert config.eval.cls_thresholds == (0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
    assert config.eval.min_areas == (0.0,)


def test_config_to_dict_keeps_current_runtime_shape():
    config = load_experiment_config("configs/baseline.yaml")
    runtime = config.to_dict()

    assert set(runtime) == {"paths", "seed", "model", "augmentation", "dataset", "train", "eval"}
    assert runtime["model"]["encoder_name"] == "pvt_v2_b2"
    assert runtime["dataset"]["image_size"] == 640
    assert "augmentations" not in runtime
    assert "img_size" not in runtime["dataset"]


def test_config_to_flat_dict_is_snapshot_friendly():
    config = load_experiment_config("configs/baseline.yaml")
    plain = config.to_flat_dict()

    assert plain["run_name"] == "pvt_v2_b2_full_frame"
    assert plain["seed"] == 42
    assert plain["encoder_name"] == "pvt_v2_b2"
    assert plain["image_size"] == 640
    assert plain["n_bins"] == 256


def test_unknown_config_key_fails_early():
    raw = load_experiment_config("configs/baseline.yaml").to_dict()
    raw["model"]["encoder"] = raw["model"].pop("encoder_name")

    with pytest.raises(ValueError, match="unknown keys in model: encoder"):
        ExperimentConfig.from_dict(raw, base_dir=Path("configs"))


def test_invalid_config_value_fails_early():
    raw = load_experiment_config("configs/baseline.yaml").to_dict()
    raw["train"]["amp"] = "mixed"

    with pytest.raises(ValueError, match="train.amp"):
        ExperimentConfig.from_dict(raw, base_dir=Path("configs"))

