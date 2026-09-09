from dataclasses import replace
import torch
from src.config import load_experiment_config
from src.training.builders import build_model
from src.losses import compute_loss


def test_rgb_only_ignores_maps():
    torch.set_num_threads(2)
    cfg = load_experiment_config('configs/baseline.yaml')
    model = build_model(replace(cfg.model, use_forensics=False), pretrained=False).eval()
    assert model.forensic_fusion is None
    image = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        a = model(image)['logits']
        b = model(image, torch.randn(2, 12, 8, 8))['logits']
    torch.testing.assert_close(a, b)
    assert model.forensic_gate_stats()['max_abs'] == 0


def test_dct_aux_trains_branch_with_closed_gates():
    torch.set_num_threads(2)
    cfg = load_experiment_config('configs/baseline.yaml')
    model = build_model(replace(cfg.model, dct_aux_weight=0.2), pretrained=False).train()
    batch = dict(mask=torch.rand(2, 1, 64, 64), label=torch.ones(2, 1))
    out = model(torch.randn(2, 3, 64, 64), torch.randn(2, 12, 8, 8))
    assert out['dct_aux_logits'].shape == (2, 1, 8, 8)
    loss = compute_loss(out, batch, dct_aux_weight=0.2)
    loss.backward()
    grad = model.forensic_fusion.branch.input_projection[0].weight.grad
    assert grad is not None and grad.abs().sum() > 0
    assert torch.isfinite(loss)
    model.eval()
    with torch.no_grad():
        assert 'dct_aux_logits' not in model(torch.randn(2, 3, 64, 64), torch.randn(2, 12, 8, 8))


def test_experiment_configs_roundtrip():
    from src.inference.submission import InferenceConfig
    baseline = load_experiment_config('configs/baseline.yaml')
    for name in ('dct_aux',):
        cfg = load_experiment_config(f'configs/{name}.yaml')
        assert cfg.train == baseline.train
        assert cfg.dataset.image_size == 640
        assert cfg.dataset == baseline.dataset
        assert cfg.augmentation == baseline.augmentation
        assert cfg.eval == baseline.eval
        assert cfg.paths.run_name != baseline.paths.run_name
        for snapshot in (cfg.to_dict(), cfg.to_flat_dict()):
            assert InferenceConfig.from_snapshot(snapshot).model == cfg.model


def test_rgb_dataset_and_predictor_without_maps(tmp_path, monkeypatch):
    import numpy as np
    import pandas as pd
    from PIL import Image
    import src.data.dataset as module
    from src.data.data_workspace import DataWorkspace
    from src.inference.predict import Predictor, ThresholdConfig
    from src.training.builders import AmpContext
    workspace = DataWorkspace(tmp_path)
    workspace.test_root.mkdir(parents=True)
    Image.fromarray(np.zeros((40, 56, 3), dtype=np.uint8)).save(workspace.test_root / 'a.jpg')
    def unexpected(*args):
        raise AssertionError('RGB-only must not extract forensic features')
    monkeypatch.setattr(module, 'forensic_maps', unexpected)
    monkeypatch.setattr(module, 'luma_qtable', unexpected)
    ds = module.AIIJCDataset(workspace, pd.DataFrame({'img_path': ['a.jpg']}),
                            False, 64, 42, mode='test', use_forensics=False)
    assert 'fmap' not in ds[0]
    cfg = load_experiment_config('configs/baseline.yaml')
    model = build_model(cfg.model, pretrained=False)
    predictor = Predictor(model, ThresholdConfig(), AmpContext(torch.device('cpu'), torch.float32, False, False))
    results = list(predictor.predict(torch.utils.data.DataLoader(ds)))
    assert results[0].mask.shape == (40, 56)
