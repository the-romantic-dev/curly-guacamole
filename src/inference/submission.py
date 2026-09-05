"""Build competition submission.csv and predictions/ from a saved run."""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.config import ModelConfig
from src.data.data_workspace import DataWorkspace
from src.data.dataset import AIIJCDataset
from src.inference.predict import Prediction, Predictor, ThresholdConfig
from src.training.builders import AmpContext, build_model
from src.training.runs import Run


class SubmissionWriter:
    def __init__(self, template: pd.DataFrame, test_rows: pd.DataFrame, output_dir: str | Path):
        self.template = template.copy()
        self.output_dir = Path(output_dir).resolve()
        self._written: set[str] = set()
        for frame, columns in ((template, ("img_path", "prediction_path")), (test_rows, ("img_path",))):
            for column in columns:
                if column not in frame or frame[column].isna().any():
                    raise ValueError(f"missing values or column: {column}")
                if not frame[column].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
                    raise ValueError(f"{column} must contain nonempty strings")
                if frame[column].duplicated().any():
                    raise ValueError(f"duplicate values in {column}")
        if set(template.img_path) != set(test_rows.img_path):
            raise ValueError("template img_path values must match test.csv exactly")
        self.paths = {row.img_path: self._output_path(row.prediction_path) for row in template.itertuples()}
        # Detect aliases, including Windows case-insensitive filenames.
        if len({str(path).casefold() for path in self.paths.values()}) != len(self.paths):
            raise ValueError("prediction paths resolve to duplicate files")

    def _output_path(self, raw: str) -> Path:
        path = PurePosixPath(raw.replace("\\", "/"))
        if (PureWindowsPath(raw).drive or path.is_absolute() or ".." in path.parts
                or len(path.parts) < 2 or path.parts[0] != "predictions"
                or path.suffix.lower() != ".png" or any(":" in part for part in path.parts)):
            raise ValueError(f"prediction path must be a relative PNG under predictions/: {raw}")
        resolved = self.output_dir.joinpath(*path.parts).resolve()
        if not resolved.is_relative_to(self.output_dir / "predictions"):
            raise ValueError(f"prediction path escapes output directory: {raw}")
        return resolved

    def write(self, predictions: Iterable[Prediction]) -> Path:
        for prediction in predictions:
            if prediction.image_path not in self.paths or prediction.image_path in self._written:
                raise ValueError(f"unexpected or duplicate prediction: {prediction.image_path}")
            mask = prediction.mask
            if mask.ndim != 2 or mask.dtype != np.uint8 or not np.isin(mask, (0, 255)).all():
                raise ValueError("prediction must be a single-channel uint8 mask containing only 0/255")
            path = self.paths[prediction.image_path]
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask).save(path, format="PNG")
            self._written.add(prediction.image_path)
        if self._written != set(self.paths):
            raise ValueError("predictions are missing for some template rows")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self.output_dir / "submission.csv"
        self.template.to_csv(csv_path, index=False)
        return csv_path


@dataclass(frozen=True)
class InferenceConfig:
    model: ModelConfig
    image_size: int
    seed: int
    data_path: Path
    device: str
    amp: str
    batch_size: int
    workers: int

    @classmethod
    def from_snapshot(cls, snapshot: dict) -> "InferenceConfig":
        """Read both the training engine's flat snapshot and nested experiment configs."""
        model = snapshot.get("model", snapshot)
        dataset = snapshot.get("dataset", snapshot)
        paths = snapshot.get("paths", snapshot)
        train = snapshot.get("train", snapshot)
        return cls(
            ModelConfig.from_dict({key: model[key] for key in (
                "encoder_name", "decoder_channels", "forensic_channels", "aux_weight", "norm",
            )}),
            int(dataset["image_size"]), int(snapshot["seed"]), Path(paths["data_path"]),
            train["device"], train["amp"], int(train["batch_size"]), int(train["workers"]),
        )


def create_submission(
    run_dir: str | Path,
    output_dir: str | Path,
    *,
    thresholds: ThresholdConfig | None = None,
    data_path: str | Path | None = None,
    template_path: str | Path | None = None,
    device: str | None = None,
) -> Path:
    """Load ckpt/best.pt (EMA preferred) and write the template-aligned submission."""
    run = Run.open(run_dir)
    config = InferenceConfig.from_snapshot(run.snapshot)
    if thresholds is None:
        best = run.summary.get("best") or {}
        keys = ("mask_threshold", "cls_threshold", "min_area")
        if not all(key in best for key in keys):
            raise ValueError("run summary has no operating point; pass thresholds explicitly")
        thresholds = ThresholdConfig(**{key: float(best[key]) for key in keys})
    workspace = DataWorkspace(Path(data_path) if data_path is not None else config.data_path)
    test_rows = workspace.test_csv
    template = pd.read_csv(template_path or workspace.test_root / "submission.csv")
    writer = SubmissionWriter(template, test_rows, output_dir)
    dataset = AIIJCDataset(workspace, test_rows, False, config.image_size, config.seed, mode="test")
    inference_device = torch.device(device or config.device)
    amp = AmpContext(inference_device, torch.bfloat16 if config.amp == "bf16" else torch.float16,
                     inference_device.type == "cuda" and config.amp != "off", False)
    checkpoint = torch.load(run.dir / "ckpt" / "best.pt", map_location="cpu", weights_only=True)
    model = build_model(config.model, pretrained=False)
    model.load_state_dict(checkpoint["ema"] if checkpoint.get("ema") is not None else checkpoint["model"])
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.workers)
    return writer.write(Predictor(model, thresholds, amp).predict(loader))
