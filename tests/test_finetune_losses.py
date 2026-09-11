import numpy as np
import pytest
import torch

from src.losses import SegmentationLoss


def test_focal_zero_equals_bce_and_easy_pixels_are_suppressed():
    from src.losses import BinaryFocalLoss
    logits = torch.tensor([[[[-50., -2., 0., 2., 50.]]]], requires_grad=True)
    target = torch.tensor([[[[0., 1., 0., 1., 0.]]]])
    torch.testing.assert_close(BinaryFocalLoss(0)(logits, target),
                               torch.nn.functional.binary_cross_entropy_with_logits(logits, target))
    result = BinaryFocalLoss(1)(logits, target)
    result.backward()
    assert torch.isfinite(result) and torch.isfinite(logits.grad).all()
    easy = torch.tensor([[[[8.]]]])
    assert BinaryFocalLoss(1)(easy, torch.ones_like(easy)) < .001 * BinaryFocalLoss(0)(easy, torch.ones_like(easy))


def test_area_dice_is_weighted_per_image_and_negative_weight_stays_one():
    target = torch.zeros(3, 1, 10, 10)
    target[0, :, 0, 0] = 1
    target[1, :, :5] = 1
    logits = torch.zeros_like(target)
    criterion = SegmentationLoss(dice_area_reference=.05, dice_area_max_weight=3.)
    actual, losses, _ = criterion._dice(logits, target, None)
    weights = torch.tensor([np.sqrt(5), 1., 1.], dtype=torch.float32)
    torch.testing.assert_close(actual, (losses*weights).sum()/weights.sum())
    valid = torch.ones_like(target)
    padded_logits = torch.nn.functional.pad(logits, (0, 10), value=100)
    padded_target = torch.nn.functional.pad(target, (0, 10), value=1)
    padded_valid = torch.nn.functional.pad(valid, (0, 10))
    torch.testing.assert_close(criterion._dice(padded_logits, padded_target, padded_valid)[0], actual)


def test_signed_distance_direction_normalization_and_empty_masks():
    from src.data.targets import SignedDistanceTarget
    mask = torch.zeros(1, 9, 9)
    mask[:, 2:7, 2:7] = 1
    distance = SignedDistanceTarget()(mask)
    assert distance[0, 4, 4] == pytest.approx(-2/np.hypot(9, 9))
    assert distance[0, 2, 2] == 0
    assert distance[0, 0, 4] == pytest.approx(2/np.hypot(9, 9))
    assert SignedDistanceTarget()(torch.zeros_like(mask)).eq(0).all()
    assert SignedDistanceTarget()(torch.ones_like(mask)).eq(0).all()
    padded = torch.nn.functional.pad(mask, (0, 3))
    valid = torch.nn.functional.pad(torch.ones_like(mask), (0, 3))
    result = SignedDistanceTarget()(padded, valid)
    torch.testing.assert_close(result[:, :, :9], distance)
    assert result[:, :, 9:].eq(0).all()


def test_boundary_gradients_aux_disabled_and_cls_retained():
    from src.data.targets import SignedDistanceTarget
    mask = torch.zeros(2, 1, 9, 9)
    mask[0, :, 2:7, 2:7] = 1
    distance = torch.stack([SignedDistanceTarget()(item) for item in mask])
    logits = torch.zeros_like(mask, requires_grad=True)
    aux = torch.zeros_like(mask, requires_grad=True)
    cls = torch.zeros(2, 1, requires_grad=True)
    criterion = SegmentationLoss(pixel_loss='focal', focal_gamma=1, boundary_weight=1,
                                 aux_weight=.4, aux_loss_weight=0)
    out = {'logits': logits, 'aux_logits': aux, 'cls_logits': cls}
    batch = {'mask': mask, 'boundary_distance': distance}
    result = criterion(out, batch)
    assert set(result.components) == {'focal', 'dice', 'boundary', 'cls'}
    grad = torch.autograd.grad(result.components['boundary'], logits, retain_graph=True)[0]
    assert grad[0, 0, 4, 4] < 0 and grad[0, 0, 0, 4] > 0
    assert grad[1].eq(0).all()
    result.total.backward()
    assert aux.grad is None and cls.grad.abs().sum() > 0
    assert torch.isfinite(logits.grad).all()
    batch['mask'].zero_(); batch['boundary_distance'].zero_()
    assert criterion(out, batch).components['boundary'].item() == 0


def test_recipe_retains_parent_architecture_and_roundtrips():
    from src.config import ExperimentConfig, load_experiment_config
    path = 'configs/jpeg576_pretrained_18ep_full_train_finetune_3ep_focal_boundary.yaml'
    cfg = load_experiment_config(path)
    parent = load_experiment_config('configs/jpeg576_pretrained_18ep_full_train_finetune_3ep.yaml')
    assert cfg.model == parent.model
    assert cfg.train == parent.train and cfg.augmentation == parent.augmentation
    assert cfg.loss.aux_loss_weight == 0 and cfg.loss.pixel_loss == 'focal'
    assert cfg.loss.boundary_weight > 0 and cfg.loss.dice_area_reference == .05
    assert ExperimentConfig.from_dict(cfg.to_dict()) == cfg


@pytest.mark.parametrize('values', [{'pixel_loss': 'bad'}, {'focal_gamma': -1},
    {'boundary_weight': float('nan')}, {'dice_area_reference': 1.1},
    {'dice_area_max_weight': .5}, {'aux_loss_weight': -1}])
def test_invalid_options(values):
    from src.config import LossConfig
    with pytest.raises(ValueError):
        LossConfig(**values)


def test_fractional_focal_gamma_has_finite_gradients_at_zero_bce():
    from src.losses import BinaryFocalLoss
    logits = torch.tensor([[[[-50., 50.]]]], requires_grad=True)
    BinaryFocalLoss(.5)(logits, torch.tensor([[[[0., 1.]]]])).backward()
    assert torch.isfinite(logits.grad).all()


def test_validation_transfers_boundary_targets():
    from dataclasses import replace
    from src.config import LossConfig, load_experiment_config
    from src.data.targets import SignedDistanceTarget
    from src.training.builders import AmpContext
    from src.training.validation import validate

    class Model(torch.nn.Module):
        def forward(self, image, fmap, **kwargs):
            return {'logits': image[:, :1], 'cls_logits': torch.zeros(len(image), 1)}

    cfg = replace(load_experiment_config('configs/baseline.yaml'),
                  loss=LossConfig(pixel_loss='focal', boundary_weight=1))
    mask = torch.zeros(2, 1, 8, 8)
    mask[0, :, 2:6, 2:6] = 1
    batch = {'image': torch.zeros(2, 3, 8, 8), 'mask': mask,
             'original_mask': [m.squeeze(0) for m in mask],
             'boundary_distance': torch.stack([SignedDistanceTarget()(m) for m in mask])}
    amp = AmpContext(torch.device('cpu'), torch.float32, False, False)
    result = validate(Model(), [batch], amp, cfg, torch.device('cpu'))
    assert 'boundary' in result.loss_components


def test_old_loss_snapshot_resumes_with_defaults_but_rejects_new_loss(tmp_path, monkeypatch):
    from dataclasses import replace
    from src.config import LossConfig, load_experiment_config
    from src.training.engine import ExperimentRunner
    from src.training.runs import Run
    import src.training.engine as engine

    cfg = load_experiment_config('configs/baseline.yaml')
    cfg = replace(cfg, paths=replace(cfg.paths, runs_path=tmp_path),
                  train=replace(cfg.train, device='cpu', amp='off', resume=True))
    snapshot = cfg.to_flat_dict()
    for key in LossConfig().to_dict():
        if key not in {'dice_scope', 'dice_weight'}:
            snapshot.pop(key)
    run = Run.create(tmp_path, cfg.paths.run_name, tensorboard=False)
    run.save_snapshot(snapshot)
    (run.dir / 'ckpt/last.pt').touch()
    class Protocol:
        def verify_run(self, saved):
            pass
    monkeypatch.setattr(engine.EvaluationProtocol, 'load', lambda path: Protocol())
    ExperimentRunner(cfg)._check_resume_protocol()
    with pytest.raises(ValueError, match='loss.pixel_loss'):
        ExperimentRunner(replace(cfg, loss=LossConfig(pixel_loss='focal')))._check_resume_protocol()
