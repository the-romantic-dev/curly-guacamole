from dataclasses import replace

import pytest
from src.config import load_experiment_config
from src.training.builders import build_model
import torch


def test_nonlinear_initialization_matches_bilinear_and_both_layers_learn():
    from src.modules.strided_resize import StridedResize
    from torch.nn import functional as F

    module = StridedResize('nonlinear')
    image = torch.rand(2, 3, 48, 64)
    expected = (F.interpolate(image, scale_factor=.5, mode='bilinear', align_corners=False)
                - module.mean) / module.std
    torch.testing.assert_close(module(image), expected, atol=1e-6, rtol=1e-5)
    module(image).square().mean().backward()
    for layer in (module.conv[0], module.conv[2]):
        assert torch.isfinite(layer.weight.grad).all()
        assert (layer.weight.grad.abs().sum(dim=(1, 2, 3)) > 0).all()


def test_nonlinear_recipe_gradients_and_reload():
    from src.inference.submission import InferenceConfig
    torch.set_num_threads(1)
    config = load_experiment_config('configs/rgb576_strided_mlp.yaml')
    assert config.model.resize_variant == 'nonlinear'
    assert config.model.norm == 'batch'
    for snapshot in (config.to_dict(), config.to_flat_dict()):
        assert InferenceConfig.from_snapshot(snapshot).model == config.model
    model = build_model(config.model, pretrained=False).train()
    output = model(torch.rand(2, 3, 128, 128))
    assert output['logits'].shape == (2, 1, 64, 64)
    (output['logits'].square().mean() + output['aux_logits'].square().mean()).backward()
    convs = [m for m in model.input_resize.modules() if isinstance(m, torch.nn.Conv2d)]
    assert [(m.in_channels, m.out_channels, m.stride) for m in convs] == [(3, 16, (1, 1)), (16, 3, (2, 2))]
    assert all(m.weight.grad is not None and m.weight.grad.abs().sum() > 0 for m in convs)
    restored = build_model(config.model, pretrained=False).eval()
    restored.load_state_dict(model.state_dict())
    model.eval()
    image = torch.rand(1, 3, 128, 128)
    with torch.no_grad():
        torch.testing.assert_close(model(image)['logits'], restored(image)['logits'])
    with pytest.raises(ValueError, match='resize_variant'):
        replace(config.model, strided_resize=False)


@pytest.mark.parametrize('nested', [False, True])
def test_resume_old_linear_snapshot_and_reject_variant_change(tmp_path, monkeypatch, nested):
    from src.training.engine import ExperimentRunner, EvaluationProtocol
    from src.training.runs import Run
    from src.inference.submission import InferenceConfig

    base = load_experiment_config('configs/rgb576_strided.yaml')
    config = replace(base, paths=replace(base.paths, runs_path=tmp_path),
                     train=replace(base.train, resume=True))
    run = Run.create(tmp_path, config.paths.run_name, tensorboard=False)
    snapshot = config.to_dict() if nested else config.to_flat_dict()
    (snapshot['model'] if nested else snapshot).pop('resize_variant')
    assert InferenceConfig.from_snapshot(snapshot).model.resize_variant == 'linear'
    run.save_snapshot(snapshot)
    (run.dir / 'ckpt' / 'last.pt').touch()
    class Protocol:
        def verify_run(self, saved):
            pass
    monkeypatch.setattr(EvaluationProtocol, 'load', lambda path: Protocol())
    ExperimentRunner(config)._check_resume_protocol()
    changed = replace(config, model=replace(config.model, resize_variant='nonlinear'))
    with pytest.raises(ValueError, match='resize_variant'):
        ExperimentRunner(changed)._check_resume_protocol()
