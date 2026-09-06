from dataclasses import replace

import pytest
import torch

from src.config import load_experiment_config
from src.decoders import create_decoder
from src.training.builders import build_model


def test_emcad_protocol_and_snapshot():
    from src.inference.submission import InferenceConfig

    config = load_experiment_config('configs/emcad_mixed_original.yaml')
    baseline = load_experiment_config('configs/baseline_mixed_original.yaml')
    assert config.model.decoder_name == 'emcad'
    assert config.paths.run_name != baseline.paths.run_name
    assert config.train == replace(baseline.train, resume=False)
    assert config.dataset == baseline.dataset
    assert config.augmentation == baseline.augmentation
    assert config.eval == baseline.eval
    for snapshot in (config.to_dict(), config.to_flat_dict()):
        assert InferenceConfig.from_snapshot(snapshot).model == config.model


def test_emcad_training_and_reload():
    from src.losses import compute_loss

    config = load_experiment_config('configs/emcad_mixed_original.yaml')
    model = build_model(config.model, pretrained=False).train()
    with torch.no_grad():
        for block in model.forensic_fusion.fusion_blocks.values():
            block.channel_gate.fill_(0.1)
    batch = dict(image=torch.randn(2, 3, 64, 96), fmap=torch.randn(2, 12, 8, 12),
                 mask=torch.ones(2, 1, 64, 96), label=torch.ones(2, 1))
    output = model(batch['image'], batch['fmap'])
    assert output['logits'].shape == output['aux_logits'].shape == batch['mask'].shape
    loss = compute_loss(output, batch, config.model.aux_weight)
    loss.backward()
    assert torch.isfinite(loss)
    for module in (model.encoder, model.forensic_fusion.branch, model.decoder):
        grads = [p.grad for p in module.parameters() if p.requires_grad]
        assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
        assert any(g.abs().sum() > 0 for g in grads)
    restored = build_model(config.model, pretrained=False).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    model.eval()
    with torch.no_grad():
        actual = restored(batch['image'], batch['fmap'])
        expected = model(batch['image'], batch['fmap'])
    assert 'aux_logits' not in actual
    torch.testing.assert_close(actual['logits'], expected['logits'])


def test_emcad_odd_feature_sizes_and_validation():
    decoder = create_decoder('emcad', encoder_channels=[8, 16, 32, 64],
                             encoder_strides=[4, 8, 16, 32]).eval()
    features = [torch.randn(2, c, h, w) for c, h, w in
                [(8, 17, 25), (16, 9, 13), (32, 5, 7), (64, 3, 4)]]
    output, aux = decoder(features)
    assert output.shape == (2, 8, 17, 25)
    assert aux is None
    with pytest.raises(ValueError, match='kernel'):
        create_decoder('emcad', encoder_channels=[8, 16, 32, 64],
                       encoder_strides=[4, 8, 16, 32], kernel_sizes=[2])


def test_emcad_640_budget():
    from src.budget import count_gflops

    config = load_experiment_config('configs/emcad_mixed_original.yaml')
    with torch.device('meta'):
        model = build_model(config.model, pretrained=False).eval()
        assert count_gflops(model, 640) <= 100
