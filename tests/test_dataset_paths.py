import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from src.data.data_workspace import DataWorkspace
from src.data.dataset import AIIJCDataset


@pytest.mark.parametrize("mode", ["train", "val", "test"])
def test_dataset_reads_paths_and_preserves_targets(tmp_path, mode):
    workspace = DataWorkspace(tmp_path)
    root = workspace.test_root if mode == "test" else workspace.train_root
    (root / "images").mkdir(parents=True)
    image = np.zeros((24, 40, 3), dtype=np.uint8)
    image[..., 2] = 255
    cv2.imencode(".png", image)[1].tofile(root / "images" / "образец.png")
    row = {"img_path" if mode == "test" else "chng_img_path": r"images\образец.png"}
    if mode != "test":
        row["gt_path"] = "mask.png"
        cv2.imencode(".png", np.full((24, 40), 255, dtype=np.uint8))[1].tofile(root / "mask.png")
    dataset = AIIJCDataset(workspace, pd.DataFrame([row]), mode == "train", 32, 42, mode=mode)

    output = dataset[0]

    assert output["image"].shape == (3, 32, 32)
    assert output["fmap"].shape == (12, 4, 4)
    assert output["image"][0].mean() > output["image"][2].mean()  # RGB conversion
    if mode == "test":
        assert "mask" not in output
        assert "label" not in output
        assert output["original_size"].tolist() == [24, 40]
        assert output["image_path"] == row["img_path"]
        batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=1)))
        assert batch["original_size"].shape == (1, 2)
    else:
        assert output["mask"].shape == (1, 32, 32)
        assert torch.all(output["mask"] == 1)
        assert output["label"].tolist() == [1.0]


@pytest.mark.parametrize("mode, row", [
    ("train", {"chng_img_path": "x.png"}),
    ("val", {"gt_path": "x.png"}),
    ("test", {"chng_img_path": "x.png"}),
])
def test_dataset_rejects_missing_columns_before_iteration(tmp_path, mode, row):
    with pytest.raises(ValueError, match="columns"):
        AIIJCDataset(DataWorkspace(tmp_path), pd.DataFrame([row]), False, 32, 42, mode=mode)


@pytest.mark.parametrize("path", [None, "", "  "])
def test_dataset_rejects_empty_image_paths(tmp_path, path):
    with pytest.raises(ValueError, match="img_path"):
        AIIJCDataset(DataWorkspace(tmp_path), pd.DataFrame([{"img_path": path}]), False, 32, 42)
