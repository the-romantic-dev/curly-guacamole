import pytest
import torch

from src.decoders import EMCADDecoder


@pytest.mark.parametrize('norm', ['batch', 'group'])
def test_emcad_custom_geometry_auxiliary_and_feature_gradients(norm):
    model = EMCADDecoder([8, 16, 32, 64], [4, 8, 16, 32], norm=norm,
                         use_aux=True, kernel_sizes=(1, 3), expansion_factor=4,
                         lgag_kernel_size=1, activation='relu6')
    features = [torch.randn(2, c, h, w, requires_grad=True)
                for c, h, w in [(8, 17, 25), (16, 9, 13), (32, 5, 7), (64, 3, 4)]]
    output, aux = model(features)
    assert model.output_stride == 4
    assert output.shape == (2, 8, 17, 25)
    assert aux.shape == (2, 1, 17, 25)
    (output.square().mean() + aux.square().mean()).backward()
    for feature in features:
        assert feature.grad is not None
        assert torch.isfinite(feature.grad).all()
        assert feature.grad.abs().sum() > 0
    model.eval()
    with torch.no_grad():
        assert model(features)[1] is None


@pytest.mark.parametrize('options,match', [
    ({'encoder_channels': [8, 16, 32]}, 'four encoder scales'),
    ({'encoder_strides': [4, 8, 16, 16]}, 'four encoder scales'),
    ({'encoder_channels': [0, 16, 32, 64]}, 'even encoder channels'),
    ({'encoder_channels': [7, 16, 32, 64]}, 'even encoder channels'),
    ({'kernel_sizes': []}, 'kernel'),
    ({'kernel_sizes': [2]}, 'kernel'),
    ({'lgag_kernel_size': 2}, 'kernel'),
    ({'expansion_factor': 0}, 'expansion_factor'),
    ({'activation': 'gelu'}, 'activation'),
    ({'norm': 'unknown'}, 'unknown'),
])
def test_emcad_rejects_invalid_geometry_and_options(options, match):
    kwargs = dict(encoder_channels=[8, 16, 32, 64], encoder_strides=[4, 8, 16, 32])
    kwargs.update(options)
    with pytest.raises(ValueError, match=match):
        EMCADDecoder(**kwargs)


def test_emcad_rejects_unknown_options_and_missing_features():
    with pytest.raises(TypeError, match='typo'):
        EMCADDecoder([8, 16, 32, 64], [4, 8, 16, 32], typo=1)
    model = EMCADDecoder([8, 16, 32, 64], [4, 8, 16, 32])
    with pytest.raises(ValueError, match='four encoder features'):
        model([])
