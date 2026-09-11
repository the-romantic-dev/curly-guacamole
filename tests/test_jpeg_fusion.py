import pytest
import torch

from src.config import ModelConfig, load_experiment_config
from src.modules.gated_fuse import GatedFuse
from src.modules.fusion_residuals import WindowCrossAttentionResidual


VARIANTS = ('local', 'spatial', 'channel_spatial', 'film', 'cross_attention')


@pytest.mark.parametrize('variant', VARIANTS)
def test_fusion_starts_as_identity_then_trains_both_inputs(variant):
    block = GatedFuse(8, 4, variant=variant)
    rgb = torch.randn(2, 8, 5, 7, requires_grad=True)
    jpeg = torch.randn(2, 4, 3, 4, requires_grad=True)
    torch.testing.assert_close(block(rgb, jpeg), rgb, rtol=0, atol=0)
    with torch.no_grad():
        block.channel_gate.fill_(0.1)
    block(rgb, jpeg).square().mean().backward()
    for value in (rgb, jpeg):
        assert torch.isfinite(value.grad).all()
        assert value.grad.abs().sum() > 0


def test_baseline_keeps_checkpoint_keys():
    block = GatedFuse(8, 4)
    assert set(block.state_dict()) == {
        'channel_gate', 'fusion.0.weight', 'fusion.1.weight', 'fusion.1.bias',
        'fusion.1.running_mean', 'fusion.1.running_var', 'fusion.1.num_batches_tracked',
    }


@pytest.mark.parametrize('variant', ('baseline', *VARIANTS))
def test_experiment_configs_share_recipe(variant):
    config = load_experiment_config(f'configs/jpeg576_fusion_{variant}_weighted_val.yaml')
    assert config.model.fusion_variant == variant
    assert config.model.jpeg_variant == 'baseline'
    assert config.eval.small_mask_weight == 1.6
    assert config.loss.dice_scope == 'positive'
    assert config.dataset.protocol_path == 'runs/validation_protocol_20260908/protocol'
    assert config.paths.run_name == f'jpeg576_fusion_{variant}_weighted_val'


def test_fusion_config_rejects_unknown_or_non_jpeg_variant():
    with pytest.raises(ValueError, match='fusion_variant'):
        ModelConfig(fusion_variant='typo')
    with pytest.raises(ValueError, match='fusion_variant'):
        ModelConfig(fusion_variant='film')


def test_partial_attention_window_matches_unpadded_reference():
    block = WindowCrossAttentionResidual(8, 4).eval()
    rgb, jpeg = torch.randn(1, 8, 2, 3), torch.randn(1, 4, 2, 3)
    q = block.query_norm(block.query(rgb).flatten(2).transpose(1, 2))
    k, v = block.key_value(jpeg).chunk(2, dim=1)
    k = block.key_norm(k.flatten(2).transpose(1, 2))
    v = v.flatten(2).transpose(1, 2)
    scores = q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5
    expected = (scores.softmax(-1) @ v).transpose(1, 2).reshape(1, 32, 2, 3)
    torch.testing.assert_close(block(rgb, jpeg), block.output(expected))


@pytest.mark.parametrize('variant', ('baseline', *VARIANTS))
def test_builder_and_snapshot_preserve_fusion_variant(variant):
    from src.config import ExperimentConfig
    from src.training.builders import build_model

    cfg = load_experiment_config(f'configs/jpeg576_fusion_{variant}_weighted_val.yaml')
    restored = ExperimentConfig.from_dict(cfg.to_dict())
    assert restored.model == cfg.model
    with torch.device('meta'):
        model = build_model(restored.model, pretrained=False)
    assert all(block.variant == variant for block in model.forensic_fusion.fusion_blocks.values())
