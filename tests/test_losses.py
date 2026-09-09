from dataclasses import replace

import pytest
import torch

from src.losses import LossMeter, SegmentationLoss, compute_loss, soft_dice_loss


def sample():
    target = torch.zeros(3, 1, 4, 4)
    target[0, :, :2] = 1
    target[1] = 1
    logits = torch.zeros_like(target, requires_grad=True)
    return {"logits": logits, "cls_logits": torch.zeros(3, 1, requires_grad=True)}, {
        "mask": target, "label": torch.tensor([[1.], [1.], [0.]])}


def test_positive_dice_excludes_negatives_but_bce_trains_them():
    out, batch = sample()
    result = SegmentationLoss(dice_scope="positive", dice_weight=.75)(out, batch)
    expected = .75 * soft_dice_loss(out["logits"][:2], batch["mask"][:2])
    torch.testing.assert_close(result.components["dice"], expected)
    result.total.backward()
    assert out["logits"].grad[2].gt(0).all()
    assert result.total.item() == pytest.approx(sum(v.item() for v in result.components.values()))


def test_empty_and_padding_only_targets_have_zero_positive_dice():
    out, batch = sample()
    batch["valid_mask"] = torch.zeros_like(batch["mask"])
    result = SegmentationLoss(dice_scope="positive")(out, batch)
    assert result.components["dice"].item() == 0
    result.total.backward()
    assert torch.isfinite(out["logits"].grad).all()
    assert out["logits"].grad.eq(0).all()


def test_all_negative_batch_has_finite_loss_and_bce_gradient():
    out, batch = sample()
    batch["mask"].zero_()
    batch["label"].zero_()
    result = SegmentationLoss(dice_scope="positive", aux_weight=.4, dct_aux_weight=.2)(
        {**out, "aux_logits": out["logits"], "dct_aux_logits": out["logits"][:, :, ::2, ::2]}, batch)
    for key in ("dice", "aux_dice", "dct_aux_dice"):
        assert result.components[key].item() == 0
    result.total.backward()
    assert out["logits"].grad.gt(0).all()


def test_legacy_wrapper_and_positive_diagnostics():
    out, batch = sample()
    result = SegmentationLoss()(out, batch)
    torch.testing.assert_close(result.total, compute_loss(out, batch))
    torch.testing.assert_close(result.components["dice"], soft_dice_loss(out["logits"], batch["mask"]))
    meter = LossMeter()
    meter.update(result, 3)
    out2 = {k: v[2:] for k, v in out.items()}
    batch2 = {k: v[2:] for k, v in batch.items()}
    meter.update(SegmentationLoss()(out2, batch2), 1)
    values = meter.compute()
    assert values["dice_pos"] == pytest.approx(soft_dice_loss(out["logits"][:2], batch["mask"][:2]).item())
    assert values["total"] == pytest.approx((3 * result.total.item() + SegmentationLoss()(out2, batch2).total.item()) / 4)


def test_loss_config_roundtrip_and_legacy_resume_guard(tmp_path):
    from src.config import ExperimentConfig, load_experiment_config
    from src.training.engine import ExperimentRunner
    from src.training.runs import Run

    old = load_experiment_config("configs/baseline.yaml")
    assert old.loss.dice_scope == "all"
    raw = old.to_dict()
    raw["loss"] = {"dice_scope": "positive", "dice_weight": .75}
    new = ExperimentConfig.from_dict(raw)
    assert ExperimentConfig.from_dict(new.to_dict()) == new
    assert new.to_flat_dict()["dice_scope"] == "positive"
    new = replace(new, paths=replace(new.paths, runs_path=tmp_path),
                  train=replace(new.train, device="cpu", amp="off", resume=True))
    run = Run.create(tmp_path, new.paths.run_name, tensorboard=False)
    run.save_snapshot(old.to_flat_dict())
    (run.dir / "ckpt" / "last.pt").touch()
    with pytest.raises(ValueError, match="loss"):
        ExperimentRunner(new)._check_resume_protocol()


@pytest.mark.parametrize("options", [{"dice_scope": "typo"}, {"dice_weight": -1}, {"dice_weight": float("nan")}])
def test_invalid_loss_settings(options):
    from src.config import ExperimentConfig, load_experiment_config
    raw = load_experiment_config("configs/baseline.yaml").to_dict()
    raw["loss"] = options
    with pytest.raises(ValueError, match="loss"):
        ExperimentConfig.from_dict(raw)


def test_positive_loss_ignores_padding_for_main_and_aux_heads():
    out, batch = sample()
    valid = torch.ones_like(batch["mask"])
    valid[:, :, :, 2:] = 0
    batch["valid_mask"] = valid
    out["aux_logits"] = out["logits"]
    out["dct_aux_logits"] = torch.zeros(3, 1, 2, 2, requires_grad=True)
    criterion = SegmentationLoss(dice_scope="positive", aux_weight=.4, dct_aux_weight=.2)
    original = criterion(out, batch)
    changed = {**out, "logits": out["logits"] + (1-valid)*100,
               "aux_logits": out["logits"] - (1-valid)*100}
    changed["dct_aux_logits"] = out["dct_aux_logits"] + (1-valid[:, :, ::2, ::2])*100
    torch.testing.assert_close(criterion(changed, batch).total, original.total)
    original.total.backward()
    assert out["dct_aux_logits"].grad[:, :, :, 0].abs().sum() > 0
    assert out["dct_aux_logits"].grad[:, :, :, 1].eq(0).all()


def test_new_ablation_config():
    from src.config import load_experiment_config
    config = load_experiment_config("configs/positive_dice.yaml")
    assert config.loss.dice_scope == "positive"
    assert config.loss.dice_weight == 1.0
    assert config.paths.run_name != load_experiment_config("configs/baseline.yaml").paths.run_name


def test_fractional_dct_validity_weights_intersection_once():
    # A one-column valid region becomes half-valid when reduced from 2x2 to 1x1.
    mask = torch.ones(1, 1, 2, 2)
    valid = torch.tensor([[[[1., 0.], [1., 0.]]]])
    out = {"logits": torch.full_like(mask, 30.), "cls_logits": torch.tensor([[30.]]),
           "dct_aux_logits": torch.tensor([[[[30.]]]], requires_grad=True)}
    result = SegmentationLoss(dice_scope="positive", dct_aux_weight=.2)(
        out, {"mask": mask, "valid_mask": valid, "label": torch.ones(1, 1)})
    assert result.components["dct_aux_dice"].item() == pytest.approx(0., abs=1e-6)


def test_legacy_scalar_matches_independent_formula():
    out, batch = sample()
    out["aux_logits"] = out["logits"] + .7
    def reference(logits):
        probs = logits.sigmoid().flatten(1)
        target = batch["mask"].flatten(1)
        return torch.nn.functional.binary_cross_entropy_with_logits(logits, batch["mask"]) + (
            1 - (2 * (probs * target).sum(1) + 1) / (probs.sum(1) + target.sum(1) + 1)).mean()
    expected = reference(out["logits"]) + .4 * reference(out["aux_logits"]) + .3 * (
        torch.nn.functional.binary_cross_entropy_with_logits(out["cls_logits"], batch["label"]))
    torch.testing.assert_close(compute_loss(out, batch, .4), expected)
