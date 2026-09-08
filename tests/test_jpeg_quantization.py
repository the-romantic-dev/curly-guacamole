import io
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.forensic.dct.jpeg import luma_qtable
from src.forensic.dct.constants import ZIGZAG


def jpeg_bytes():
    buffer = io.BytesIO()
    rgb = np.random.default_rng(4).integers(0, 256, (32, 48, 3), dtype=np.uint8)
    Image.fromarray(rgb).save(buffer, format='JPEG', qtables=[list(range(1, 65))])
    return buffer.getvalue()


def test_known_quantization_table_and_legacy_compatibility(tmp_path):
    data = jpeg_bytes()
    path = tmp_path / 'image.jpg'
    path.write_bytes(data)
    expected = np.arange(1, 65, dtype=np.float32).reshape(8, 8)
    for source in (data, path):
        np.testing.assert_array_equal(luma_qtable(source, order='natural'), expected)
        np.testing.assert_array_equal(luma_qtable(source, order='legacy_zigzag'), expected.ravel()[ZIGZAG])
    np.testing.assert_array_equal(luma_qtable(data), expected.ravel()[ZIGZAG])
    with pytest.raises(ValueError, match='order'):
        luma_qtable(data, order='typo')


def test_recompression_uses_selected_table_order():
    from src.data.augmentation.transforms.random_jpeg_recompression import RandomJPEGRecompression
    rgb = np.asarray(Image.open(io.BytesIO(jpeg_bytes())).convert('RGB'))
    natural = RandomJPEGRecompression((70, 71), 1, jpeg_qtable_order='natural')
    legacy = RandomJPEGRecompression((70, 71), 1, jpeg_qtable_order='legacy_zigzag')
    image, table = natural.jpeg_recompression(rgb, np.random.default_rng(42))
    other_image, other_table = legacy.jpeg_recompression(rgb, np.random.default_rng(42))
    import cv2
    ok, encoded = cv2.imencode('.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 70])
    assert ok
    expected = np.asarray(Image.open(io.BytesIO(encoded.tobytes())).quantization[0]).reshape(8, 8)
    np.testing.assert_array_equal(table, expected)
    np.testing.assert_array_equal(other_table, expected.ravel()[ZIGZAG])
    np.testing.assert_array_equal(image, other_image)


def test_config_snapshot_and_new_experiment():
    from src.config import load_experiment_config, ExperimentConfig
    from src.inference.submission import InferenceConfig
    old = load_experiment_config('configs/baseline_mixed_original_long.yaml')
    new = load_experiment_config('configs/baseline_mixed_original_natural_q.yaml')
    assert old.dataset.jpeg_qtable_order == 'legacy_zigzag'
    assert new.dataset.jpeg_qtable_order == 'natural'
    assert new.dataset.image_size == 640 and new.train.epochs == 18
    assert replace(new.dataset, jpeg_qtable_order='legacy_zigzag') == old.dataset
    for attr in ('train', 'model', 'augmentation', 'loss', 'eval', 'seed'):
        assert getattr(old, attr) == getattr(new, attr)
    assert old.paths.run_name != new.paths.run_name
    for snapshot in (new.to_dict(), new.to_flat_dict()):
        assert InferenceConfig.from_snapshot(snapshot).jpeg_qtable_order == 'natural'
    legacy = old.to_dict()
    legacy['dataset'].pop('jpeg_qtable_order')
    assert InferenceConfig.from_snapshot(legacy).jpeg_qtable_order == 'legacy_zigzag'
    assert ExperimentConfig.from_dict(legacy).dataset.jpeg_qtable_order == 'legacy_zigzag'
    legacy['dataset']['jpeg_qtable_order'] = 'typo'
    with pytest.raises(ValueError, match='jpeg_qtable_order'):
        ExperimentConfig.from_dict(legacy)


def test_dataset_builders_use_protocol_for_original_and_recompressed_jpeg(tmp_path):
    from src.config import load_experiment_config
    from src.data.data_workspace import DataWorkspace
    from src.data.data_sample import DataSample
    from src.data.augmentation.base import AugmentationStage
    from src.training.builders import build_datasets
    from src.forensic.dct import forensic_maps
    cfg = load_experiment_config('configs/baseline_mixed_original_natural_q.yaml')
    workspace = DataWorkspace(tmp_path)
    workspace.train_root.mkdir()
    image_path = workspace.train_root / 'image.jpg'
    image_path.write_bytes(jpeg_bytes())
    Image.new('L', (48, 32), 255).save(workspace.train_root / 'mask.png')
    rows = pd.DataFrame({'chng_img_path':['image.jpg'], 'gt_path':['mask.png'], 'is_negative':[False]})
    cfg = replace(cfg, dataset=replace(cfg.dataset, image_size=32),
                  augmentation=replace(cfg.augmentation, jpeg_recompression_probability=1))
    train, val = build_datasets(cfg, workspace, rows, rows)
    assert train.jpeg_qtable_order == val.jpeg_qtable_order == 'natural'
    rgb = val.load_image(image_path)
    q = np.arange(1,65,dtype=np.float32).reshape(8,8)
    sample = DataSample(image=rgb, qtable=q, fmap=forensic_maps(rgb,q))
    expected = val.preprocessor.to_output(val.preprocessor.resize(sample))['fmap']
    np.testing.assert_allclose(val[0]['fmap'].numpy(), expected.numpy())
    transform = train.augmentations.pipeline[AugmentationStage.BEFORE_FORENSICS][0]
    assert transform.jpeg_qtable_order == 'natural'


def test_resume_rejects_changed_qtable_protocol(tmp_path):
    import yaml
    from src.config import load_experiment_config
    from src.training.engine import ExperimentRunner
    cfg = load_experiment_config('configs/baseline_mixed_original_natural_q.yaml')
    cfg = replace(cfg, paths=replace(cfg.paths, runs_path=tmp_path), train=replace(cfg.train, device='cpu'))
    run_dir = tmp_path / cfg.paths.run_name
    (run_dir / 'ckpt').mkdir(parents=True)
    (run_dir / 'ckpt/last.pt').touch()
    snapshot = cfg.to_flat_dict()
    snapshot.pop('jpeg_qtable_order')
    (run_dir / 'config.yaml').write_text(yaml.safe_dump(snapshot), encoding='utf-8')
    with pytest.raises(ValueError, match='jpeg_qtable_order'):
        ExperimentRunner(cfg)._check_resume_protocol()
