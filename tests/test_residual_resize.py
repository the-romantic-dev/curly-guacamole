from src.config import load_experiment_config
from src.training.builders import build_model
import pytest
import torch
import torch.nn.functional as F


@pytest.mark.parametrize('variant', ['residual_paper', 'residual_compact'])
def test_residual_resize_starts_with_random_correction_and_learns(variant):
    from src.modules.strided_resize import StridedResize

    torch.set_num_threads(1)
    module = StridedResize(variant).train()
    image = torch.rand(2, 3, 64, 80)
    expected = (F.interpolate(image, scale_factor=.5, mode='bilinear', align_corners=False)
                - module.mean) / module.std
    assert not torch.allclose(module(image), expected)
    module(image).square().mean().backward()
    assert module.conv.output.weight.grad.abs().sum() > 0
    assert module.conv.stem[0].weight.grad.abs().sum() > 0
    module.eval()
    # Removing the correction leaves exactly the bilinear skip.
    with torch.no_grad():
        module.conv.output.weight.zero_()
        module.conv.output.bias.zero_()
        torch.testing.assert_close(module(image), expected)


@pytest.mark.parametrize('suffix', ['paper', 'compact'])
def test_residual_recipe_checkpoint_and_budget(suffix):
    from src.inference.submission import InferenceConfig
    from src.budget import count_gflops

    config = load_experiment_config(f'configs/rgb576_residual_{suffix}.yaml')
    for snapshot in (config.to_dict(), config.to_flat_dict()):
        assert InferenceConfig.from_snapshot(snapshot).model == config.model
    with torch.device('meta'):
        meta = build_model(config.model, pretrained=False)
    assert count_gflops(meta, 576) < 100
    model = build_model(config.model, pretrained=False).train()
    out = model(torch.rand(2, 3, 128, 128))
    assert out['logits'].shape == out['aux_logits'].shape == (2, 1, 64, 64)
    (out['logits'].square().mean() + out['aux_logits'].square().mean()).backward()
    assert model.input_resize.conv.output.weight.grad.abs().sum() > 0
    clone = build_model(config.model, pretrained=False).eval()
    clone.load_state_dict(model.state_dict())
    model.eval()
    image = torch.rand(1, 3, 128, 128)
    with torch.no_grad():
        torch.testing.assert_close(model(image)['logits'], clone(image)['logits'])
