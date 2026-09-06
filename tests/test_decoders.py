import pytest
import torch


def test_registry_discovery_and_errors():
    from src.decoders import create_decoder, is_decoder, list_decoders

    assert list_decoders() == ['segformer', 'unet']
    assert list_decoders('seg*') == ['segformer']
    assert is_decoder('unet')
    with pytest.raises(ValueError, match='Unknown decoder'):
        create_decoder('missing', encoder_channels=[4], encoder_strides=[4])
    with pytest.raises(TypeError):
        create_decoder('segformer', encoder_channels=[4], encoder_strides=[4], typo=1)


@pytest.mark.parametrize('name,kwargs,stride', [('unet', {'decoder_channels': [16, 8, 4, 4, 4]}, 1), ('segformer', {'embed_dim': 8}, 4)])
def test_decoder_contract(name, kwargs, stride):
    from src.decoders import Decoder, create_decoder

    model = create_decoder(name, encoder_channels=[8, 16, 32, 64],
                           encoder_strides=[4, 8, 16, 32], use_aux=True, **kwargs)
    assert isinstance(model, Decoder)
    features = [torch.randn(2, c, 64 // s, 96 // s, requires_grad=True)
                for c, s in zip([8, 16, 32, 64], [4, 8, 16, 32])]
    output, aux = model(features)
    assert model.output_stride == stride
    assert output.shape == (2, model.out_channels, 64 // stride, 96 // stride)
    assert aux is not None
    output.mean().backward()
    assert all(f.grad is not None for f in features)


def test_custom_decoder_works_through_config_and_segmenter(monkeypatch):
    from src.decoders import Decoder, register_decoder
    from src.decoders import registry
    from src.config import ModelConfig, load_experiment_config
    from src.training.builders import build_model

    monkeypatch.setattr(registry, '_DECODERS', registry._DECODERS.copy())

    @register_decoder('custom')
    class CustomDecoder(Decoder):
        head_kernel_size = 1

        def __init__(self, encoder_channels, encoder_strides, norm, use_aux, width=7):
            super().__init__()
            self.out_channels = width
            self.output_stride = encoder_strides[0]
            self.projection = torch.nn.Conv2d(encoder_channels[0], width, 1)

        def forward(self, features):
            return self.projection(features[0]), None

    data = load_experiment_config('configs/baseline_mixed_original.yaml').model.to_dict()
    data.update(decoder_name='custom', decoder_kwargs={'width': 9})
    config = ModelConfig.from_dict(data)
    assert ModelConfig.from_dict(config.to_dict()) == config
    model = build_model(config, pretrained=False)
    assert model.decoder.out_channels == 9
    assert model.segmentation_head.kernel_size == (1, 1)
    with pytest.raises(ValueError, match='already registered'):
        register_decoder('custom')(CustomDecoder)
