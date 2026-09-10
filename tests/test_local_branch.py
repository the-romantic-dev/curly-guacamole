from dataclasses import replace

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from src.config import ModelConfig, load_experiment_config
from src.data.data_workspace import DataWorkspace
from src.data.dataset import AIIJCDataset
from src.training.builders import build_model


def test_native_residual_energy_survives_reduction():
    from src.data.local_preprocess import LocalPreprocessor

    image = np.zeros((64, 96, 3), dtype=np.uint8)
    image[:, ::2] = 255
    prepared = LocalPreprocessor(32)(image)
    assert prepared.shape == (15, 32, 32)
    assert torch.isfinite(prepared).all()
    # Alternating signed differences cancel on reduction, their energy must not.
    assert prepared[9:].abs().mean() > 0.05
    constant = LocalPreprocessor(32)(np.full_like(image, 127))
    assert torch.count_nonzero(constant[3:]) == 0


@pytest.mark.parametrize('mode', ['train', 'val', 'test'])
def test_dataset_local_view_preserves_geometry(tmp_path, mode):
    workspace = DataWorkspace(tmp_path)
    root = workspace.test_root if mode == 'test' else workspace.train_root
    root.mkdir(parents=True)
    image = np.zeros((48, 80, 3), dtype=np.uint8)
    image[12:36, 40:72] = 255
    cv2.imencode('.png', image)[1].tofile(root / 'image.png')
    cv2.imencode('.png', image[..., 0])[1].tofile(root / 'mask.png')
    row = {'img_path': 'image.png'} if mode == 'test' else {
        'chng_img_path': 'image.png', 'gt_path': 'mask.png'}
    ds = AIIJCDataset(workspace, pd.DataFrame([row]), mode == 'train', 32, 42,
                     mode=mode, local_image_size=64, original_targets=mode == 'val')
    sample = ds[0]
    assert sample['local_input'].shape == (15, 64, 64)
    local = torch.nn.functional.interpolate(sample['local_input'][None, :3], (32, 32), mode='area')[0]
    assert ((local[0] > 0) == (sample['image'][0] > 0)).float().mean() > .95
    if mode == 'val':
        assert sample['original_mask'].shape == (48, 80)


def local_config(size=128):
    config = load_experiment_config('configs/baseline.yaml')
    return replace(config.model, local_image_size=size)


@pytest.mark.parametrize('shape', [(64, 96), (123, 171), (23, 91)])
def test_local_preprocessing_matches_original_features(shape):
    from src.data.local_preprocess import LocalPreprocessor
    from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

    image = np.random.default_rng(7).integers(0, 256, (*shape, 3), dtype=np.uint8)
    prep = LocalPreprocessor(64)
    rgb = image.astype(np.float32) / 255.
    ycc = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    dx = np.diff(ycc, axis=1, append=ycc[:, -1:])
    dy = np.diff(ycc, axis=0, append=ycc[-1:])
    rgb = (prep._resize(rgb) - np.asarray(IMAGENET_DEFAULT_MEAN, np.float32)) / np.asarray(IMAGENET_DEFAULT_STD, np.float32)
    expected = np.concatenate([rgb, prep._resize(dx), prep._resize(dy),
                               prep._resize(abs(dx)), prep._resize(abs(dy))], axis=2).transpose(2, 0, 1)
    np.testing.assert_array_equal(prep(image).numpy(), expected)


def test_local_loaders_bound_prefetch_and_validation_batch(tmp_path):
    from src.training.builders import build_datasets, build_loaders
    cfg = load_experiment_config('configs/local.yaml')
    rows = pd.DataFrame({'chng_img_path': ['a.jpg', 'b.jpg'], 'gt_path': ['a.png', 'b.png'],
                         'is_negative': [False, True]})
    datasets = build_datasets(cfg, DataWorkspace(tmp_path), rows, rows)
    train, val = build_loaders(replace(cfg.train, workers=10, batch_size=8), *datasets)
    assert train.prefetch_factor == val.prefetch_factor == 1
    assert val.batch_size == train.batch_size == 8


def test_local_training_independent_supervision_and_reload():
    from src.losses import SegmentationLoss

    torch.set_num_threads(1)
    torch.manual_seed(42)
    config = local_config()
    model = build_model(config, pretrained=False).train()
    image = torch.randn(2, 3, 64, 64)
    fmap = torch.randn(2, 12, 8, 8)
    local = torch.randn(2, 15, 128, 128)
    out = model(image, fmap, local_input=local)
    target = torch.zeros(2, 1, 64, 64)
    target[0, :, 20:28, 36:44] = 1
    batch = {'mask': target, 'label': torch.tensor([[1.], [0.]])}
    assert out['logits'].shape == out['aux_logits'].shape == target.shape
    # Local supervision has a gradient path without global encoder/decoder evidence.
    torch.nn.functional.binary_cross_entropy_with_logits(out['aux_logits'], target).backward(retain_graph=True)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.local_branch.parameters())
    assert all(p.grad is None for p in model.encoder.parameters())
    model.zero_grad(set_to_none=True)
    loss = SegmentationLoss(dice_scope='positive', aux_weight=.4)(out, batch).total
    loss.backward()
    for module in (model.encoder, model.decoder, model.local_branch):
        grads = [p.grad for p in module.parameters() if p.requires_grad]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        assert any(g.abs().sum() > 0 for g in grads)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.step()
    restored = build_model(config, pretrained=False).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    model.eval()
    with torch.no_grad():
        expected = model(image, fmap, local_input=local)
        actual = restored(image, fmap, local_input=local)
    assert 'aux_logits' not in actual
    torch.testing.assert_close(actual['logits'], expected['logits'])
    with pytest.raises(ValueError, match='local_input'):
        restored(image, fmap)


def test_local_snapshot_and_budget():
    from src.budget import count_gflops
    from src.inference.submission import InferenceConfig

    base = load_experiment_config('configs/baseline.yaml')
    config = replace(base, model=local_config(1024))
    for snapshot in (config.to_dict(), config.to_flat_dict()):
        assert InferenceConfig.from_snapshot(snapshot).model == config.model
    with torch.device('meta'):
        model = build_model(config.model, pretrained=False).eval()
        baseline = build_model(base.model, pretrained=False).eval()
        assert count_gflops(baseline, 640) < count_gflops(model, 640) <= 100


@pytest.mark.parametrize('size', [-1, 31, True])
def test_invalid_local_size(size):
    data = load_experiment_config('configs/baseline.yaml').model.to_dict()
    data['local_image_size'] = size
    with pytest.raises(ValueError, match='local_image_size'):
        ModelConfig.from_dict(data)


def test_budget_counts_inference_without_changing_batchnorm():
    from src.budget import count_gflops

    model = build_model(local_config(128), pretrained=False).train()
    before = {name: tensor.clone() for name, tensor in model.named_buffers()}
    training_budget = count_gflops(model, 64)
    assert model.training
    for name, tensor in model.named_buffers():
        torch.testing.assert_close(tensor, before[name])
    assert training_budget == pytest.approx(count_gflops(model.eval(), 64))


def test_augmented_views_and_mask_remain_aligned(tmp_path):
    from src.data.augmentation.pipeline import AugmentationPipeline

    workspace = DataWorkspace(tmp_path)
    root = workspace.train_root
    root.mkdir(parents=True)
    image = np.zeros((128, 192, 3), dtype=np.uint8)
    image[32:96, 80:160] = 255
    cv2.imencode('.png', image)[1].tofile(root / 'image.png')
    cv2.imencode('.png', image[..., 0])[1].tofile(root / 'mask.png')
    base = load_experiment_config('configs/baseline.yaml')
    augmentation = replace(base.augmentation, jpeg_recompression_probability=0,
                           final_full_frame_epochs=0, crop_scale_range=(.35, .7))
    ds = AIIJCDataset(workspace, pd.DataFrame([dict(chng_img_path='image.png', gt_path='mask.png')]),
                     True, 64, 42, augmentations=AugmentationPipeline(augmentation), local_image_size=128)
    for _ in range(8):
        sample = ds[0]
        local = torch.nn.functional.interpolate(sample['local_input'][None, :3], (64, 64), mode='area')[0]
        for view in (sample['image'], local):
            assert ((view[0] > 0) == sample['mask'][0].bool()).float().mean() > .95


def test_training_validation_and_submission_use_local_input(tmp_path):
    from src.inference.predict import Predictor, ThresholdConfig
    from src.training.builders import build_amp, build_optimizer, build_scheduler, build_ema
    from src.training.engine import train_one_epoch
    from src.training.validation import validate

    base = load_experiment_config('configs/local.yaml')
    config = replace(base, model=local_config(128),
                     train=replace(base.train, device='cpu', amp='off', workers=0, accum_steps=1),
                     eval=replace(base.eval, mask_thresholds=(.5,), cls_thresholds=(0.,), min_areas=(0.,)))
    model = build_model(config.model, pretrained=False)
    amp = build_amp(config.train)
    optimizer = build_optimizer(config.train, model)
    target = torch.zeros(2, 1, 64, 64)
    target[0, :, 20:28, 36:44] = 1
    batch = dict(image=torch.randn(2, 3, 64, 64), fmap=torch.randn(2, 12, 8, 8),
                 local_input=torch.randn(2, 15, 128, 128), mask=target,
                 label=torch.tensor([[1.], [0.]]), original_mask=[target[0, 0].bool(), target[1, 0].bool()],
                 original_size=torch.tensor([[64, 64], [64, 64]]), image_path=['a.jpg', 'b.jpg'])
    trained = train_one_epoch(model=model, loader=[batch], optimizer=optimizer,
                              scheduler=build_scheduler(config.train, optimizer, 1), scaler=amp.scaler(),
                              ema=build_ema(config.train, model), amp=amp, config=config, device=amp.device)
    assert trained.seen == 2 and np.isfinite(trained.loss)
    assert trained.loss_components['aux_bce'] > 0
    validation = validate(model, [batch], amp, config, amp.device)
    predictions = list(Predictor(model, ThresholdConfig(), amp).predict([batch]))
    from src.training.metric import score_masks
    direct = score_masks([p.mask for p in predictions], [m.numpy() for m in batch['original_mask']])
    assert validation.tuned.dice_pos == pytest.approx(direct.dice_pos)
    assert validation.tuned.fpr_neg == direct.fpr_neg


def test_resume_rejects_different_local_resolution(tmp_path):
    from src.training.engine import ExperimentRunner
    from src.training.runs import Run

    base = load_experiment_config('configs/local.yaml')
    config = replace(base, paths=replace(base.paths, runs_path=tmp_path),
                     train=replace(base.train, resume=True))
    run = Run.create(tmp_path, config.paths.run_name, tensorboard=False)
    snapshot = config.to_flat_dict()
    snapshot['local_image_size'] = 960
    run.save_snapshot(snapshot)
    (run.dir / 'ckpt' / 'last.pt').touch()
    with pytest.raises(ValueError, match='local_image_size'):
        ExperimentRunner(config)._check_resume_protocol()
    assert run.snapshot['local_image_size'] == 960


def test_local_config_rejects_letterbox():
    from src.config import ExperimentConfig

    raw = load_experiment_config('configs/local.yaml').to_dict()
    raw['dataset']['resize_mode'] = 'letterbox'
    with pytest.raises(ValueError, match='stretch'):
        ExperimentConfig.from_dict(raw)


@pytest.mark.parametrize('branch', ['local', 'luma'])
def test_saved_local_model_submission_matches_checkpoint_evaluation(tmp_path, monkeypatch, branch):
    from PIL import Image

    from src.eval.checkpoints import CheckpointEvaluator
    from src.inference.submission import create_submission
    from src.training.metric import score_masks
    from src.training.runs import Run

    torch.set_num_threads(1)
    base = load_experiment_config(f'configs/{branch}.yaml')
    branch_config = local_config(128) if branch == 'local' else ModelConfig(luma_image_size=128)
    config = replace(base, model=branch_config,
                     paths=replace(base.paths, data_path=tmp_path / 'data'),
                     dataset=replace(base.dataset, image_size=64),
                     train=replace(base.train, device='cpu', amp='off', batch_size=2, workers=0))
    workspace = DataWorkspace(config.paths.data_path)
    for root in (workspace.train_root, workspace.test_root):
        root.mkdir(parents=True)
    rows, masks = [], []
    for i, (h, w) in enumerate(((73, 101), (96, 64))):
        image = np.random.default_rng(i).integers(0, 256, (h, w, 3), dtype=np.uint8)
        mask = np.zeros((h, w), dtype=np.uint8)
        if i == 0:
            mask[20:30, 50:60] = 255
        for root in (workspace.train_root, workspace.test_root):
            Image.fromarray(image).save(root / f'{i}.jpg')
        Image.fromarray(mask).save(workspace.train_root / f'{i}.png')
        rows.append(dict(chng_img_path=f'{i}.jpg', gt_path=f'{i}.png', group_id=str(i),
                         domain='plain', target_kind='provided'))
        masks.append(mask)
    pd.DataFrame({'img_path': ['0.jpg', '1.jpg']}).to_csv(workspace.test_root / 'test.csv', index=False)
    pd.DataFrame({'img_path': ['0.jpg', '1.jpg'], 'prediction_path': ['predictions/0.png', 'predictions/1.png']}).to_csv(
        workspace.test_root / 'submission.csv', index=False)
    model = build_model(config.model, pretrained=False).eval()
    run = Run.create(tmp_path / 'runs', 'local', tensorboard=False)
    run.save_snapshot(config.to_flat_dict())
    run.save_summary({'best': dict(mask_threshold=.5, cls_threshold=0., min_area=0.)})
    run.save_state({'model': model.state_dict()}, 'best.pt')
    output = create_submission(run.dir, tmp_path / 'submission', data_path=config.paths.data_path)
    predictions = [np.asarray(Image.open(output.parent / f'predictions/{i}.png')) for i in range(2)]
    assert [p.shape for p in predictions] == [m.shape for m in masks]
    direct = score_masks(predictions, masks)
    monkeypatch.setenv('AIIJC_DATA_PATH', str(config.paths.data_path))
    evaluated = CheckpointEvaluator(run.dir, device='cpu', batch_size=2, workers=0).evaluate(
        pd.DataFrame(rows), tmp_path / 'evaluation', purpose='synthetic-smoke')
    assert evaluated['combined']['dice_pos'] == pytest.approx(direct.dice_pos)
    assert evaluated['combined']['fpr_neg'] == direct.fpr_neg
