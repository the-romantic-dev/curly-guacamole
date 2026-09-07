from pathlib import Path

import pytest

from src.config import ExperimentConfig, load_experiment_config


def test_portable_config_uses_machine_paths(tmp_path, monkeypatch):
    import yaml

    raw = load_experiment_config("configs/baseline.yaml").to_dict()
    raw["paths"] = {"run_name": "portable"}
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    for machine in ("local", "server"):
        data_path = tmp_path / machine / "data"
        runs_path = tmp_path / machine / "runs"
        monkeypatch.setenv("AIIJC_DATA_PATH", str(data_path))
        monkeypatch.setenv("AIIJC_RUNS_PATH", str(runs_path))
        config = load_experiment_config(config_path)
        assert config.paths.data_path == data_path
        assert config.paths.runs_path == runs_path
        assert config.paths.run_name == "portable"


def test_saved_paths_do_not_override_machine_settings(tmp_path, monkeypatch):
    raw = load_experiment_config("configs/baseline.yaml").to_dict()
    monkeypatch.setenv("AIIJC_DATA_PATH", str(tmp_path / "data"))
    monkeypatch.setenv("AIIJC_RUNS_PATH", str(tmp_path / "runs"))
    config = ExperimentConfig.from_dict(raw)
    assert config.paths.data_path == tmp_path / "data"
    assert config.paths.runs_path == tmp_path / "runs"


def test_machine_defaults_are_independent_of_yaml_location_and_cwd(tmp_path, monkeypatch):
    import global_config

    raw = load_experiment_config("configs/baseline.yaml").to_dict()
    raw["paths"] = {"run_name": "portable"}
    monkeypatch.delenv("AIIJC_DATA_PATH", raising=False)
    monkeypatch.delenv("AIIJC_RUNS_PATH", raising=False)
    monkeypatch.setattr(global_config, "DATA_PATH", "data")
    monkeypatch.setattr(global_config, "RUNS_PATH", "runs")
    monkeypatch.setattr(global_config, "PROJECT_ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    config = ExperimentConfig.from_dict(raw, base_dir=tmp_path)
    assert config.paths.data_path == global_config.PROJECT_ROOT / "data"
    assert config.paths.runs_path == global_config.PROJECT_ROOT / "runs"


def test_experiment_yamls_have_no_machine_paths():
    import yaml

    for path in Path("configs").glob("*.yaml"):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert set(raw["paths"]) == {"run_name"}, path
        load_experiment_config(path)


def test_dotenv_paths_reload_and_environment_takes_priority(tmp_path, monkeypatch):
    import global_config

    config_path = Path("configs/baseline.yaml").resolve()
    monkeypatch.setattr(global_config, "PROJECT_ROOT", tmp_path)
    monkeypatch.delenv("AIIJC_DATA_PATH", raising=False)
    monkeypatch.delenv("AIIJC_RUNS_PATH", raising=False)
    dotenv = tmp_path / ".env"
    dotenv.write_text('AIIJC_DATA_PATH="my data"\nAIIJC_RUNS_PATH=outputs\n', encoding="utf-8")
    config = load_experiment_config(config_path)
    assert config.paths.data_path == tmp_path / "my data"
    assert config.paths.runs_path == tmp_path / "outputs"
    dotenv.write_text("AIIJC_DATA_PATH=new_data\n", encoding="utf-8")
    monkeypatch.setenv("AIIJC_RUNS_PATH", str(tmp_path / "override"))
    config = load_experiment_config(config_path)
    assert config.paths.data_path == tmp_path / "new_data"
    assert config.paths.runs_path == tmp_path / "override"


def test_load_baseline_config(monkeypatch):
    monkeypatch.setenv("AIIJC_DATA_PATH", "D:/Challenges/AIIJC2026/data")
    monkeypatch.setenv("AIIJC_RUNS_PATH", str(Path("runs").resolve()))
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
