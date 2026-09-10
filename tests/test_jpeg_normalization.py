import copy

import numpy as np  # noqa: F401 -- initialize NumPy before torch on Windows.
import torch
from torch import nn

from src.modules.jpeg_branch import JPEGBranch


def test_jpeg_eval_matches_training_without_updating_checkpoint_buffers():
    torch.manual_seed(42)
    model = JPEGBranch((8, 12, 16))
    reference = copy.deepcopy(model).train()
    # Old checkpoints retain all BatchNorm running buffers and load strictly.
    model.load_state_dict(reference.state_dict(), strict=True)
    model.eval()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    samples = [dict(bins=torch.randint(0, 21, (64, 96), dtype=torch.uint8),
                    qtable=torch.full((8, 8), q), geometry=(0, 0, 64, 96, 0, 0, 0))
               for q in (1., 40.)]
    sizes = {8: (8, 12), 16: (4, 6), 32: (2, 3)}
    with torch.no_grad():
        expected = reference(samples, sizes)
        actual = model(samples, sizes)
    for stride in sizes:
        torch.testing.assert_close(actual[stride], expected[stride])
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    assert all(not layer.training for layer in model.modules())


def test_jpeg_training_bn_matches_standard_bn_outputs_gradients_and_buffers():
    model = JPEGBranch((8, 12, 16))
    for layer in model.modules():
        if not isinstance(layer, nn.BatchNorm2d):
            continue
        standard = nn.BatchNorm2d(layer.num_features, eps=layer.eps, momentum=layer.momentum)
        standard.load_state_dict(layer.state_dict(), strict=True)
        x = torch.randn(1, layer.num_features, 4, 6, requires_grad=True)
        y = x.detach().clone().requires_grad_()
        actual, expected = layer(x), standard(y)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.square().sum().backward()
        expected.square().sum().backward()
        torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)
        for key, value in layer.state_dict().items():
            torch.testing.assert_close(value, standard.state_dict()[key], rtol=0, atol=0)
