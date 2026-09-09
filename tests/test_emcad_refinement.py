import pytest
import torch

from src.modules.segmenter import Segmenter


@pytest.mark.parametrize('name', ['stride4', 'stride2_rgb'])
def test_experiment_protocol_and_inference(name):
    from src.config import load_experiment_config
    from src.inference.submission import InferenceConfig

    config = load_experiment_config(f'configs/{name}.yaml')
    parent = load_experiment_config('configs/positive_dice.yaml')
    assert config.paths.run_name != parent.paths.run_name
    assert config.train.batch_size * config.train.accum_steps == 16
    for field in ('dataset', 'augmentation', 'eval', 'seed'):
        assert getattr(config, field) == getattr(parent, field)
    for snapshot in (config.to_dict(), config.to_flat_dict()):
        assert InferenceConfig.from_snapshot(snapshot).model == config.model


@pytest.mark.parametrize('options', [
    {'output_refinement_channels': 144},
    {'rgb_refinement_channels': 32, 'rgb_detail_channels': 24},
])
def test_refinement_receives_segmentation_gradients_and_reloads(options):
    model = Segmenter('pvt_v2_b2', pretrained=False,
                      decoder_kwargs=options, aux_weight=0.4).train()
    image = torch.randn(2, 3, 64, 96)
    output = model(image)
    assert output['logits'].shape == output['aux_logits'].shape == (2, 1, 64, 96)
    output['logits'].square().mean().backward()
    refinement = (model.decoder.output_refinement if 'output_refinement_channels' in options
                  else model.decoder.rgb_refinement)
    for parameter in refinement.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    assert any(p.grad.abs().sum() > 0 for p in refinement.parameters())
    restored = Segmenter('pvt_v2_b2', pretrained=False,
                         decoder_kwargs=options, aux_weight=0.4).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    model.eval()
    with torch.no_grad():
        expected = model(image)['logits']
        actual = restored(image)
    assert 'aux_logits' not in actual
    torch.testing.assert_close(actual['logits'], expected)


def test_rgb_refinement_uses_image_features_and_coarse_logits():
    from src.decoders.emcad import EMCADDecoder

    decoder = EMCADDecoder([8, 16, 32, 64], [4, 8, 16, 32],
                           rgb_refinement_channels=8, rgb_detail_channels=4).eval()
    image = torch.randn(2, 3, 65, 97, requires_grad=True)
    features = torch.randn(2, 8, 17, 25, requires_grad=True)
    coarse = torch.randn(2, 1, 17, 25, requires_grad=True)
    logits = decoder.refine_logits(image, features, coarse)
    assert logits.shape == (2, 1, 33, 49)
    logits.square().mean().backward()
    for tensor in (image, features, coarse):
        assert tensor.grad is not None and tensor.grad.abs().sum() > 0


@pytest.mark.parametrize('options', [
    {'output_refinement_channels': -1},
    {'rgb_refinement_channels': 2.5},
    {'rgb_refinement_channels': 8, 'rgb_detail_channels': 0},
    {'output_refinement_channels': 8, 'rgb_refinement_channels': 8},
])
def test_invalid_refinement_options(options):
    from src.decoders.emcad import EMCADDecoder

    with pytest.raises(ValueError):
        EMCADDecoder([8, 16, 32, 64], [4, 8, 16, 32], **options)
