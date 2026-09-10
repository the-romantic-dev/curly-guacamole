from dataclasses import replace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.config import ModelConfig, load_experiment_config
from src.data.data_sample import DataSample
from src.data.preprocess import SamplePreprocessor


def test_strided_resize_starts_as_bilinear_and_learns():
    from src.modules.strided_resize import StridedResize

    module = StridedResize()
    image = torch.rand(2, 3, 64, 80)
    mean = torch.tensor([.485, .456, .406])[None, :, None, None]
    std = torch.tensor([.229, .224, .225])[None, :, None, None]
    expected = (F.interpolate(image, scale_factor=.5, mode='bilinear', align_corners=False) - mean) / std
    torch.testing.assert_close(module(image), expected)
    module(image).square().mean().backward()
    assert module.conv.weight.grad.abs().sum() > 0


def test_preprocessor_keeps_high_resolution_rgb_and_working_targets():
    image = np.random.default_rng(7).integers(0, 256, (96, 128, 3), dtype=np.uint8)
    sample = DataSample(image=image, mask=np.ones((96, 128), np.float32),
                        fmap=np.zeros((1, 12, 16), np.float32))
    preprocessor = SamplePreprocessor(32, strided_resize=True)
    output = preprocessor.to_output(preprocessor.resize(sample))
    assert output['image'].shape == (3, 64, 64)
    assert output['mask'].shape == (1, 32, 32)
    assert 0 <= output['image'].min() <= output['image'].max() <= 1


def test_recipe_and_snapshot():
    from src.inference.submission import InferenceConfig

    config = load_experiment_config('configs/rgb576_strided.yaml')
    assert config.dataset.image_size == 576
    assert config.model.strided_resize and not config.model.use_forensics
    for snapshot in (config.to_dict(), config.to_flat_dict()):
        assert InferenceConfig.from_snapshot(snapshot).model.strided_resize
    with pytest.raises(ValueError, match='strided_resize'):
        replace(config.model, use_forensics=True)
    with pytest.raises(ValueError, match='strided_resize'):
        replace(config, dataset=replace(config.dataset, resize_mode='letterbox'))


def test_model_output_optimizer_and_checkpoint():
    from src.training.builders import build_model, build_optimizer
    from src.config import TrainConfig

    torch.set_num_threads(1)
    config = ModelConfig(use_forensics=False, strided_resize=True)
    model = build_model(config, pretrained=False).train()
    output = model(torch.rand(2, 3, 128, 128))
    assert output['logits'].shape == (2, 1, 64, 64)
    assert output['aux_logits'].shape == (2, 1, 64, 64)
    output['logits'].square().mean().backward()
    assert model.input_resize.conv.weight.grad.abs().sum() > 0
    optimizer = build_optimizer(TrainConfig(), model)
    assert any(p is model.input_resize.conv.weight for g in optimizer.param_groups for p in g['params'])
    restored = build_model(config, pretrained=False).eval()
    restored.load_state_dict(model.state_dict())
    model.eval()
    image = torch.rand(1, 3, 128, 128)
    with torch.no_grad():
        torch.testing.assert_close(model(image)['logits'], restored(image)['logits'])


def test_training_validation_prediction_and_dataset(tmp_path):
    import cv2
    import pandas as pd
    from src.data.data_workspace import DataWorkspace
    from src.data.collation import ValidationCollator
    from src.training.builders import (build_datasets, build_model, build_amp, build_optimizer,
                                       build_scheduler, build_ema)
    from src.training.engine import train_one_epoch
    from src.training.validation import validate
    from src.inference.predict import Predictor, ThresholdConfig
    from src.training.metric import score_masks

    base = load_experiment_config('configs/rgb576_strided.yaml')
    config = replace(base, dataset=replace(base.dataset, image_size=64),
                     train=replace(base.train, device='cpu', amp='off', workers=0, accum_steps=1),
                     eval=replace(base.eval, mask_thresholds=(.5,), cls_thresholds=(0.,), min_areas=(0.,)))
    workspace = DataWorkspace(tmp_path)
    workspace.train_root.mkdir(parents=True)
    rows = []
    for i, shape in enumerate([(96, 160), (128, 80)]):
        image = np.random.default_rng(i).integers(0, 256, (*shape, 3), dtype=np.uint8)
        mask = np.zeros(shape, np.uint8)
        if i == 0:
            mask[20:40, 20:40] = 255
        cv2.imencode('.png', image)[1].tofile(workspace.train_root / f'{i}.png')
        cv2.imencode('.png', mask)[1].tofile(workspace.train_root / f'{i}_mask.png')
        rows.append(dict(chng_img_path=f'{i}.png', gt_path=f'{i}_mask.png', is_negative=bool(i)))
    train, val = build_datasets(config, workspace, pd.DataFrame(rows), pd.DataFrame(rows))
    batch = ValidationCollator()([train[0], train[1]])
    assert batch['image'].shape == (2, 3, 128, 128)
    assert batch['mask'].shape == (2, 1, 64, 64)
    assert 'fmap' not in batch
    model = build_model(config.model, pretrained=False)
    amp = build_amp(config.train)
    optimizer = build_optimizer(config.train, model)
    trained = train_one_epoch(model=model, loader=[batch], optimizer=optimizer,
                              scheduler=build_scheduler(config.train, optimizer, 1), scaler=amp.scaler(),
                              ema=build_ema(config.train, model), amp=amp, config=config, device=amp.device)
    assert trained.seen == 2 and np.isfinite(trained.loss)
    batch = ValidationCollator()([val[0], val[1]])
    batch['original_size'] = torch.tensor([[96, 160], [128, 80]])
    batch['image_path'] = ['0.png', '1.png']
    validation = validate(model, [batch], amp, config, amp.device)
    predictions = list(Predictor(model, ThresholdConfig(), amp).predict([batch]))
    assert [p.mask.shape for p in predictions] == [(96, 160), (128, 80)]
    direct = score_masks([p.mask for p in predictions], [m.numpy() for m in batch['original_mask']])
    assert validation.tuned.dice_pos == pytest.approx(direct.dice_pos)
    assert validation.tuned.fpr_neg == direct.fpr_neg


@pytest.mark.parametrize('nested', [False, True])
def test_old_resume_defaults_to_disabled(tmp_path, monkeypatch, nested):
    from src.training.engine import ExperimentRunner, EvaluationProtocol
    from src.training.runs import Run

    base = load_experiment_config('configs/rgb576.yaml')
    config = replace(base, paths=replace(base.paths, runs_path=tmp_path),
                     train=replace(base.train, resume=True))
    run = Run.create(tmp_path, config.paths.run_name, tensorboard=False)
    snapshot = config.to_dict() if nested else config.to_flat_dict()
    (snapshot['model'] if nested else snapshot).pop('strided_resize')
    run.save_snapshot(snapshot)
    (run.dir / 'ckpt' / 'last.pt').touch()
    class Protocol:
        def verify_run(self, saved):
            pass
    monkeypatch.setattr(EvaluationProtocol, 'load', lambda path: Protocol())
    ExperimentRunner(config)._check_resume_protocol()
    changed = replace(config, model=replace(config.model, strided_resize=True))
    with pytest.raises(ValueError, match='strided_resize'):
        ExperimentRunner(changed)._check_resume_protocol()


def test_budget_counts_double_size_input():
    from src.budget import count_gflops
    from src.training.builders import build_model

    with torch.device('meta'):
        baseline = build_model(ModelConfig(use_forensics=False), pretrained=False)
        strided = build_model(ModelConfig(use_forensics=False, strided_resize=True), pretrained=False)
    baseline_cost = count_gflops(baseline, 576)
    strided_cost = count_gflops(strided, 576)
    assert strided_cost - baseline_cost == pytest.approx(2 * 576 * 576 * 3 * 3 * 3 * 3 / 1e9)
    assert strided_cost < 100
