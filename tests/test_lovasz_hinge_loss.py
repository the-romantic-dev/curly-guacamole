import pytest
import torch

from src.config import ExperimentConfig, load_experiment_config
from src.losses import SegmentationLoss, lovasz_hinge_loss


def evaluate(target, logits, **options):
    out = {'logits': logits, 'cls_logits': torch.zeros(len(target), 1)}
    return SegmentationLoss(aux_weight=0, **options)(out, {'mask': target})


def test_lovasz_hinge_matches_the_hand_computed_jaccard_extension():
    # Flattened: labels [1, 1, 0, 0], logits [2, -1, .5, -3].
    # Hinge errors [-1, 2, 1.5, -2]; sorted [2, 1.5, -1, -2] with labels [1, 0, 1, 0].
    # Jaccard along the sorted prefix: [.5, 2/3, 1, 1] -> increments [.5, 1/6, 1/3, 0].
    # Loss = 2 * .5 + 1.5 / 6 = 1.25.
    target = torch.tensor([[[[1., 1.], [0., 0.]]]])
    logits = torch.tensor([[[[2., -1.], [.5, -3.]]]])

    assert lovasz_hinge_loss(logits, target).item() == pytest.approx(1.25)


def test_lovasz_hinge_is_zero_once_every_pixel_clears_the_unit_margin():
    target = torch.zeros(1, 1, 8, 8)
    target[..., :4, :] = 1
    logits = torch.where(target > .5, 1., -1.)

    assert lovasz_hinge_loss(logits, target).item() == 0


def test_lovasz_hinge_averages_images_and_charges_a_negative_for_its_strongest_false_positive():
    target = torch.zeros(2, 1, 4, 4)
    target[0, :, :2, :] = 1
    logits = torch.full_like(target, -3.)
    logits[0] = torch.where(target[0] > .5, 5., -5.)   # perfect positive image
    logits[1, 0, 3, 3] = .5                              # one confident pixel on a negative

    assert lovasz_hinge_loss(logits, target).item() == pytest.approx((0 + 1.5) / 2)


def test_lovasz_hinge_binarizes_soft_targets_at_half():
    hard = torch.tensor([[[[1., 0.], [0., 0.]]]])
    soft = torch.tensor([[[[.75, .25], [.1, 0.]]]])
    logits = torch.tensor([[[[-.5, .25], [-2., -2.]]]])

    torch.testing.assert_close(lovasz_hinge_loss(logits, soft), lovasz_hinge_loss(logits, hard))


def test_lovasz_component_is_weighted_and_disabled_loss_is_exactly_legacy():
    target = torch.zeros(2, 1, 16, 16)
    target[0, :, 4:9, 3:11] = 1
    logits = torch.randn_like(target, requires_grad=True)

    legacy = evaluate(target, logits)
    disabled = evaluate(target, logits, lovasz_weight=0)
    active = evaluate(target, logits, lovasz_weight=.5)

    assert 'lovasz' not in disabled.components
    assert torch.equal(legacy.total, disabled.total)
    torch.testing.assert_close(active.components['lovasz'], .5 * lovasz_hinge_loss(logits, target))
    active.total.backward()
    assert torch.isfinite(logits.grad).all()


def test_lovasz_gradient_lands_on_the_mistaken_pixels_not_the_far_background():
    target = torch.zeros(1, 1, 8, 8)
    target[..., 2:5, 2:5] = 1
    logits = torch.where(target > .5, -1., 1.).requires_grad_(True)  # everything wrong, equal margins

    lovasz_hinge_loss(logits, target).backward()

    assert logits.grad[..., 2:5, 2:5].lt(0).all()   # fill the missed object
    assert logits.grad[target <= .5].gt(0).any()    # remove false positives
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize('options', [{'lovasz_weight': -1}, {'lovasz_weight': float('nan')}])
def test_invalid_lovasz_weight_rejected_by_config_and_loss(options):
    raw = load_experiment_config('configs/baseline.yaml').to_dict()
    raw['loss'].update(options)
    with pytest.raises(ValueError, match='lovasz|weights'):
        ExperimentConfig.from_dict(raw)
    with pytest.raises(ValueError, match='lovasz|weights'):
        SegmentationLoss(**options)


def test_lovasz_finetune_config_matches_the_boundary_control_except_for_the_new_term():
    candidate = load_experiment_config('configs/experiments/disentangle_b2_li760_r8_lovasz_ft.yaml')
    control = load_experiment_config('configs/experiments/disentangle_b2_li760_r8_boundary_control_ft.yaml')
    assert candidate.loss.lovasz_weight == .5
    assert control.loss.lovasz_weight == 0
    a, b = candidate.to_dict(), control.to_dict()
    a.pop('run_name'); b.pop('run_name')
    a['loss']['lovasz_weight'] = 0
    assert a == b
    assert ExperimentConfig.from_dict(candidate.to_dict()) == candidate
