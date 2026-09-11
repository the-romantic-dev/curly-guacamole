import numpy as np  # noqa: F401 - load the conda NumPy runtime before torch
import pytest
import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode

from src.modules.jpeg_branch import JPEGArtifactModule


def test_jpeg_tail_disables_cudnn_for_backward_only_locally():
    class RecordConvolutionBackend(TorchDispatchMode):
        def __init__(self):
            self.flags = []

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func in (torch.ops.aten.convolution.default,
                        torch.ops.aten.convolution_backward.default):
                self.flags.append(torch.backends.cudnn.enabled)
            return func(*args, **(kwargs or {}))

    stem = JPEGArtifactModule()
    x = torch.randn(1, 64, 16, 24, requires_grad=True)
    with torch.backends.cudnn.flags(enabled=True):
        with RecordConvolutionBackend() as record:
            y = stem.dc_layer1_tail[0](x)
            y.sum().backward()
        assert len(record.flags) >= 2
        assert not any(record.flags)
        assert torch.backends.cudnn.enabled is True
        with RecordConvolutionBackend() as other:
            stem.dc_layer0_dil[0](torch.randn(1, 21, 16, 24)).sum().backward()
        assert all(other.flags)
        assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('amp', [False, True])
def test_jpeg_tail_preserves_weights_outputs_and_gradients(device, amp):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    reference = nn.Conv2d(64, 4, 1, bias=False).to(device)
    layer = JPEGArtifactModule().dc_layer1_tail[0].to(device)
    layer.load_state_dict(reference.state_dict(), strict=True)
    reference.load_state_dict(layer.state_dict(), strict=True)
    x = torch.randn(1, 64, 16, 24, device=device, requires_grad=True)
    expected_x = x.detach().clone().requires_grad_()
    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=amp):
        with torch.backends.cudnn.flags(enabled=False):
            expected = reference(expected_x)
        actual = layer(x)
    grad = torch.randn_like(actual)
    actual.backward(grad)
    with torch.backends.cudnn.flags(enabled=False):
        expected.backward(grad)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(x.grad, expected_x.grad)
    torch.testing.assert_close(layer.weight.grad, reference.weight.grad)
