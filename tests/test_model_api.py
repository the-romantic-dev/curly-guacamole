import pytest


def test_fusion_rejects_missing_stride_and_misaligned_metadata():
    from src.modules.forensic_fusion import ForensicFusion

    with pytest.raises(ValueError, match='missing fusion strides'):
        ForensicFusion([4, 8, 16], [4, 8, 16], (4, 8, 16))
    with pytest.raises(ValueError, match='same length'):
        ForensicFusion([4, 8, 16, 32], [4, 8, 16], (4, 8, 16))
    with pytest.raises(ValueError, match='unique'):
        ForensicFusion([8, 8, 16, 32], [4, 8, 16, 32], (4, 8, 16))


def test_segmenter_gate_stats_use_public_fusion_api():
    import torch

    from src.modules.forensic_fusion import ForensicFusion
    from src.modules.segmenter import Segmenter

    model = Segmenter.__new__(Segmenter)
    torch.nn.Module.__init__(model)
    model.forensic_fusion = ForensicFusion([8, 16, 32], [4, 8, 16], (4, 8, 16))
    with torch.no_grad():
        model.forensic_fusion.fusion_blocks['16'].channel_gate.fill_(-0.75)
    assert model.forensic_gate_stats()['max_abs'] == pytest.approx(0.75)
    assert model.FORENSIC_INPUT_CHANNELS == 12
    assert model.forensic_fusion.branch.input_norm.num_features == 12


def test_default_segmenter_uses_emcad_and_preserves_state_names():
    import torch
    from src.decoders import EMCADDecoder
    from src.modules.segmenter import Segmenter

    model = Segmenter(pretrained=False, use_forensics=False).eval()
    assert isinstance(model.decoder, EMCADDecoder)
    with torch.no_grad():
        output = model(torch.randn(1, 3, 64, 96))
    assert output['logits'].shape == (1, 1, 64, 96)
    assert output['cls_logits'].shape == (1, 1)
    assert 'aux_logits' not in output
    keys = model.state_dict()
    for prefix in ('encoder.', 'decoder.', 'segmentation_head.', 'classification_head.'):
        assert any(key.startswith(prefix) for key in keys)


@pytest.mark.parametrize('option,value', [
    ('decoder_name', 'unet'), ('decoder_channels', [16, 8]), ('decoder_embed_dim', 8),
])
def test_segmenter_rejects_obsolete_decoder_arguments(option, value):
    from src.modules.segmenter import Segmenter

    with pytest.raises(TypeError, match=option):
        Segmenter('pvt_v2_b2', pretrained=False, **{option: value})
