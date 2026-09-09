import pytest
import torch

from src.config import load_experiment_config
from src.decoders import EMCADDecoder
from src.modules.segmenter import Segmenter


def test_emcad_protocol_and_snapshot():
    from src.inference.submission import InferenceConfig

    config = load_experiment_config('configs/baseline.yaml')
    assert config.model.encoder_name == 'pvt_v2_b2'
    for snapshot in (config.to_dict(), config.to_flat_dict()):
        assert InferenceConfig.from_snapshot(snapshot).model == config.model


def test_emcad_training_and_reload():
    from src.losses import compute_loss

    model = Segmenter(pretrained=False, aux_weight=0.4).train()
    with torch.no_grad():
        for block in model.forensic_fusion.fusion_blocks.values():
            block.channel_gate.fill_(0.1)
    batch = dict(image=torch.randn(2, 3, 64, 96), fmap=torch.randn(2, 12, 8, 12),
                 mask=torch.ones(2, 1, 64, 96), label=torch.ones(2, 1))
    output = model(batch['image'], batch['fmap'])
    assert output['logits'].shape == output['aux_logits'].shape == batch['mask'].shape
    loss = compute_loss(output, batch, model.aux_weight)
    loss.backward()
    assert torch.isfinite(loss)
    for module in (model.encoder, model.forensic_fusion.branch, model.decoder):
        grads = [p.grad for p in module.parameters() if p.requires_grad]
        assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
        assert any(g.abs().sum() > 0 for g in grads)
    restored = Segmenter(pretrained=False, aux_weight=0.4).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    model.eval()
    with torch.no_grad():
        actual = restored(batch['image'], batch['fmap'])
        expected = model(batch['image'], batch['fmap'])
    assert 'aux_logits' not in actual
    torch.testing.assert_close(actual['logits'], expected['logits'])


def test_emcad_odd_feature_sizes_and_validation():
    decoder = EMCADDecoder(encoder_channels=[8, 16, 32, 64],
                             encoder_strides=[4, 8, 16, 32]).eval()
    features = [torch.randn(2, c, h, w) for c, h, w in
                [(8, 17, 25), (16, 9, 13), (32, 5, 7), (64, 3, 4)]]
    output, aux = decoder(features)
    assert output.shape == (2, 8, 17, 25)
    assert aux is None
    with pytest.raises(ValueError, match='kernel'):
        EMCADDecoder(encoder_channels=[8, 16, 32, 64],
                       encoder_strides=[4, 8, 16, 32], kernel_sizes=[2])


@pytest.mark.parametrize('options', [{}, {'output_refinement_channels': 144},
                                     {'rgb_refinement_channels': 32, 'rgb_detail_channels': 24}])
def test_emcad_640_budget(options):
    from src.budget import count_gflops

    with torch.device('meta'):
        model = Segmenter(pretrained=False, decoder_kwargs=options).eval()
        assert count_gflops(model, 640) <= 100
