from dataclasses import replace

import torch

from src.config import load_experiment_config
from src.inference.submission import InferenceConfig
from src.training.builders import build_model


def test_segformer_config_and_inference_roundtrip():
    baseline = load_experiment_config('configs/baseline_mixed_original.yaml')
    config = load_experiment_config('configs/segformer_mixed_original.yaml')
    assert config.model.decoder_name == 'segformer'
    assert config.model.decoder_kwargs == {'embed_dim': 128}
    assert config.augmentation == baseline.augmentation
    assert config.dataset == baseline.dataset
    assert config.train == replace(baseline.train, resume=False)
    assert config.eval == baseline.eval
    for snapshot in (config.to_dict(), config.to_flat_dict()):
        assert InferenceConfig.from_snapshot(snapshot).model == config.model
    assert baseline.model.decoder_name == 'unet'


def test_segformer_training_and_checkpoint_roundtrip():
    from src.losses import compute_loss

    torch.set_num_threads(2)
    config = load_experiment_config('configs/segformer_mixed_original.yaml')
    model = build_model(config.model, pretrained=False).train()
    batch = dict(image=torch.randn(2, 3, 64, 96), fmap=torch.randn(2, 12, 8, 12),
                 mask=torch.ones(2, 1, 64, 96), label=torch.ones(2, 1))
    output = model(batch['image'], batch['fmap'])
    assert output['logits'].shape == output['aux_logits'].shape == batch['mask'].shape
    loss = compute_loss(output, batch, config.model.aux_weight)
    loss.backward()
    assert torch.isfinite(loss)
    for projection in model.decoder.projections:
        assert projection.weight.grad is not None
        assert torch.isfinite(projection.weight.grad).all()
        assert projection.weight.grad.abs().sum() > 0
    restored = build_model(config.model, pretrained=False).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    model.eval()
    with torch.no_grad():
        expected = model(batch['image'], batch['fmap'])
        actual = restored(batch['image'], batch['fmap'])
    assert 'aux_logits' not in actual
    torch.testing.assert_close(actual['logits'], expected['logits'])
