from dataclasses import replace

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from src.config import ModelConfig, load_experiment_config
from src.data.collation import ValidationCollator
from src.data.data_workspace import DataWorkspace
from src.data.dataset import AIIJCDataset
from src.training.builders import build_model
from src.training.transfer import BatchTransfer


@pytest.mark.parametrize('nested', [False, True])
def test_resume_snapshot_before_luma_field(tmp_path, monkeypatch, nested):
    from src.training.engine import ExperimentRunner, EvaluationProtocol
    from src.training.runs import Run

    base = load_experiment_config('configs/baseline.yaml')
    config = replace(base, paths=replace(base.paths, runs_path=tmp_path),
                     train=replace(base.train, resume=True))
    run = Run.create(tmp_path, config.paths.run_name, tensorboard=False)
    snapshot = config.to_dict() if nested else config.to_flat_dict()
    (snapshot['model'] if nested else snapshot).pop('luma_image_size')
    run.save_snapshot(snapshot)
    (run.dir / 'ckpt' / 'last.pt').touch()
    verified = []
    class Protocol:
        def verify_run(self, saved):
            verified.append(saved)
    monkeypatch.setattr(EvaluationProtocol, 'load', lambda path: Protocol())
    ExperimentRunner(config)._check_resume_protocol()
    assert len(verified) == 1
    changed = replace(config, model=replace(config.model, luma_image_size=128))
    with pytest.raises(ValueError, match='luma_image_size'):
        ExperimentRunner(changed)._check_resume_protocol()


def test_luma_native_input_and_variable_size_collation(tmp_path):
    workspace = DataWorkspace(tmp_path)
    workspace.train_root.mkdir(parents=True)
    rows = []
    for i, shape in enumerate([(48, 80), (73, 55)]):
        image = np.zeros((*shape, 3), np.uint8)
        image[12:24, 20:32] = 255
        cv2.imencode('.png', image)[1].tofile(workspace.train_root / f'{i}.png')
        rows.append(dict(chng_img_path=f'{i}.png', gt_path=f'{i}.png', is_negative=False))
    dataset = AIIJCDataset(workspace, pd.DataFrame(rows), False, 32, 42,
                          mode='val', use_forensics=False, luma_image_size=64)
    samples = [dataset[0], dataset[1]]
    assert samples[0]['native_rgb'].shape == (3, 48, 80)
    assert samples[0]['native_rgb'].dtype == torch.uint8
    assert 'local_input' not in samples[0]
    batch = ValidationCollator()(samples)
    assert [tuple(x.shape) for x in batch['native_rgb']] == [(3, 48, 80), (3, 73, 55)]
    transferred = BatchTransfer('cpu')(batch)
    assert transferred['native_rgb'][0].dtype == torch.uint8


def test_luma_preprocessing_uses_native_detail():
    from src.modules.luma_branch import LumaPreprocessor

    prep = LumaPreprocessor(64)
    source = torch.zeros(3, 64, 64, dtype=torch.uint8)
    source[:, :, ::2] = 255
    y = prep([source])
    assert y.shape == (1, 1, 64, 64)
    torch.testing.assert_close(y[0, 0, :, ::2], torch.ones(64, 32))
    assert y[0, 0, :, 1::2].count_nonzero() == 0


def test_luma_auxiliary_gradients_and_reload():
    torch.set_num_threads(1)
    config = ModelConfig(luma_image_size=128)
    model = build_model(config, pretrained=False).train()
    image = torch.randn(2, 3, 64, 64)
    fmap = torch.randn(2, 12, 8, 8)
    native = [torch.randint(256, (3, h, w), dtype=torch.uint8) for h, w in [(100, 150), (84, 111)]]
    out = model(image, fmap, native_rgb=native)
    assert out['aux_logits'].shape == out['logits'].shape == (2, 1, 64, 64)
    out['aux_logits'].square().mean().backward()
    assert all(p.grad is None for p in model.encoder.parameters())
    assert model.luma_branch.stem[0].weight.grad.abs().sum() > 0
    model.zero_grad(set_to_none=True)
    model(image, fmap, native_rgb=native)['logits'].square().mean().backward()
    assert model.luma_branch.gamma.grad is not None
    assert any(p.grad is not None for p in model.encoder.parameters())
    restored = build_model(config, pretrained=False).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(image, fmap, native_rgb=native)['logits'],
                                   restored(image, fmap, native_rgb=native)['logits'])
    with pytest.raises(ValueError, match='native_rgb'):
        restored(image, fmap)


@pytest.mark.parametrize('kwargs', [dict(luma_image_size=-1), dict(luma_image_size=33),
                                  dict(luma_image_size=True), dict(luma_image_size=128, local_image_size=128)])
def test_luma_invalid_config(kwargs):
    with pytest.raises(ValueError, match='luma'):
        ModelConfig(**kwargs)


def test_luma_snapshots_budget_and_loaders(tmp_path):
    from src.budget import count_gflops
    from src.inference.submission import InferenceConfig
    from src.training.builders import build_datasets, build_loaders

    config = load_experiment_config('configs/luma.yaml')
    for snapshot in (config.to_dict(), config.to_flat_dict()):
        assert InferenceConfig.from_snapshot(snapshot).model == config.model
    rows = pd.DataFrame(dict(chng_img_path=['a.png', 'b.png'], gt_path=['a.png', 'b.png'],
                             is_negative=[False, True]))
    train, val = build_datasets(config, DataWorkspace(tmp_path), rows, rows)
    loaders = build_loaders(replace(config.train, workers=2, batch_size=2), train, val)
    assert all(isinstance(loader.collate_fn, ValidationCollator) for loader in loaders)
    assert all(loader.prefetch_factor == 1 and loader.batch_size == 2 for loader in loaders)
    with torch.device('meta'):
        model = build_model(config.model, pretrained=False)
    cost = count_gflops(model, 640)
    assert 89.56 < cost < 100
    assert count_gflops(model, 640, native_size=(1536, 2048)) < 100


def test_luma_training_validation_and_prediction():
    from src.training.builders import build_amp, build_optimizer, build_scheduler, build_ema
    from src.training.engine import train_one_epoch
    from src.training.validation import validate
    from src.inference.predict import Predictor, ThresholdConfig
    from src.training.metric import score_masks

    base = load_experiment_config('configs/luma.yaml')
    config = replace(base, model=replace(base.model, luma_image_size=128),
                     train=replace(base.train, device='cpu', amp='off', workers=0, accum_steps=1),
                     eval=replace(base.eval, mask_thresholds=(.5,), cls_thresholds=(0.,), min_areas=(0.,)))
    model = build_model(config.model, pretrained=False)
    amp = build_amp(config.train)
    optimizer = build_optimizer(config.train, model)
    target = torch.zeros(2, 1, 64, 64)
    target[0, :, 20:28, 36:44] = 1
    batch = dict(image=torch.randn(2, 3, 64, 64), fmap=torch.randn(2, 12, 8, 8),
                 native_rgb=[torch.randint(256, (3, h, w), dtype=torch.uint8) for h,w in [(96,128),(80,111)]],
                 mask=target, label=torch.tensor([[1.], [0.]]),
                 original_mask=[target[0,0].bool(),target[1,0].bool()],
                 original_size=torch.tensor([[64,64],[64,64]]), image_path=['a.png','b.png'])
    trained = train_one_epoch(model=model, loader=[batch], optimizer=optimizer,
                              scheduler=build_scheduler(config.train, optimizer, 1), scaler=amp.scaler(),
                              ema=build_ema(config.train, model), amp=amp, config=config, device=amp.device)
    assert trained.seen == 2 and np.isfinite(trained.loss)
    assert trained.loss_components['aux_bce'] > 0
    validation = validate(model, [batch], amp, config, amp.device)
    predictions = list(Predictor(model, ThresholdConfig(), amp).predict([batch]))
    direct = score_masks([p.mask for p in predictions], [m.numpy() for m in batch['original_mask']])
    assert validation.tuned.dice_pos == pytest.approx(direct.dice_pos)
    assert validation.tuned.fpr_neg == direct.fpr_neg
