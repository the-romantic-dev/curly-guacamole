import cv2
import numpy as np
import pandas as pd


def test_build_metadata_supports_zero_workers(tmp_path, monkeypatch):
    from src.data.data_workspace import DataWorkspace
    from src.eval.metadata import build_metadata

    train_root = tmp_path / "train_stage1"
    image_rel = "stage1/train/img/coco_sample_powerpaint.jpg"
    mask_rel = "stage1/train/mask/coco_sample_powerpaint.png"

    (train_root / "stage1" / "train" / "img").mkdir(parents=True)
    (train_root / "stage1" / "train" / "mask").mkdir(parents=True)

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[:2, :2] = 255
    ok, encoded_image = cv2.imencode(".jpg", image)
    assert ok
    encoded_image.tofile(str(train_root / image_rel))
    ok, encoded_mask = cv2.imencode(".png", mask)
    assert ok
    encoded_mask.tofile(str(train_root / mask_rel))

    pd.DataFrame(
        [
            {
                "chng_img_path": image_rel,
                "gt_path": mask_rel,
                "orgl_img_path": image_rel,
            }
        ]
    ).to_csv(train_root / "stage1" / "train.csv", index=False)

    metadata = build_metadata(DataWorkspace(tmp_path), workers=0)

    assert len(metadata) == 1
    assert metadata.loc[0, "height"] == 8
    assert metadata.loc[0, "width"] == 8
    assert metadata.loc[0, "mask_area"] == 0.0625
    assert bool(metadata.loc[0, "is_negative"]) is False

    # A new workspace must reuse persisted metadata without probing any files.
    import src.eval.metadata as module
    with monkeypatch.context() as patch:
        def unexpected_probe(*args, **kwargs):
            raise AssertionError("Cached images must not be read again")
        patch.setattr(module, "_probe_sample", unexpected_probe)
        cached = build_metadata(DataWorkspace(tmp_path), workers=0)
    pd.testing.assert_frame_equal(metadata, cached)

    # An explicit refresh picks up replaced masks at the same path.
    mask[:] = 0
    ok, encoded_mask = cv2.imencode(".png", mask)
    assert ok
    encoded_mask.tofile(str(train_root / mask_rel))
    refreshed = build_metadata(DataWorkspace(tmp_path), workers=0, force_rebuild=True)
    assert refreshed.loc[0, "mask_area"] == 0

    # Changing the input table invalidates the cache and preserves row order.
    csv_path = train_root / "stage1" / "train.csv"
    table = pd.read_csv(csv_path)
    pd.concat([table, table], ignore_index=True).to_csv(csv_path, index=False)
    rebuilt = build_metadata(DataWorkspace(tmp_path), workers=0)
    assert len(rebuilt) == 2
