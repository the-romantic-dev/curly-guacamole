import io

import numpy as np
import pytest
import torch
from PIL import Image

from src.forensic.jpeg_input import JPEGInput
from src.modules.jpeg_branch import JPEGArtifactModule


def jpeg_bytes(shape=(19, 27), value=128, progressive=False):
    stream = io.BytesIO()
    Image.fromarray(np.full((*shape, 3), value, np.uint8)).save(
        stream, format='JPEG', quality=90, progressive=progressive)
    return stream.getvalue()


@pytest.mark.parametrize('progressive', [False, True])
def test_real_native_coefficients_and_quantization(progressive):
    data = JPEGInput.read(jpeg_bytes(progressive=progressive))
    assert data.bins.shape == (24, 32)
    assert data.bins.dtype == np.uint8
    assert not data.bins.any()  # Constant 128 has exactly zero centered DCT.
    assert data.geometry == (0, 0, 19, 27, 0, 0, 0)
    with Image.open(io.BytesIO(jpeg_bytes(progressive=progressive))) as image:
        np.testing.assert_array_equal(data.qtable, np.array(image.quantization[0]).reshape(8, 8))


def test_invalid_jpeg_raises_without_terminating_process():
    with pytest.raises(ValueError, match='JPEG'):
        JPEGInput.read(b'not a JPEG')


def test_large_jpeg_does_not_require_temporary_backing_files():
    native = JPEGInput.read(jpeg_bytes((1024, 1024)))
    assert native.bins.shape == (1024, 1024)


def test_module_q_weighting_and_gradients():
    torch.set_num_threads(1)
    module = JPEGArtifactModule().train()
    bins = torch.randint(21, (2, 16, 24), dtype=torch.uint8)
    qtable = torch.full((2, 8, 8), 2.)
    result = module(bins, qtable)
    assert result.shape == (2, 512, 2, 3)
    torch.testing.assert_close(result[:, 256:], 2 * result[:, :256])
    result.square().mean().backward()
    assert module.dc_layer0_dil[0].weight.grad.abs().sum() > 0


def test_dataset_collation_transfer_and_recompression(tmp_path):
    import pandas as pd

    from src.data.augmentation.transforms.random_jpeg_recompression import RandomJPEGRecompression
    from src.data.collation import ValidationCollator
    from src.data.data_sample import DataSample
    from src.data.data_workspace import DataWorkspace
    from src.data.dataset import AIIJCDataset
    from src.training.transfer import BatchTransfer

    workspace = DataWorkspace(tmp_path)
    workspace.train_root.mkdir()
    rows = []
    for i, shape in enumerate(((35, 51), (40, 64))):
        (workspace.train_root / f'{i}.jpg').write_bytes(jpeg_bytes(shape, value=200))
        rows.append(dict(chng_img_path=f'{i}.jpg', gt_path='', target_kind='original_zero'))
    dataset = AIIJCDataset(workspace, pd.DataFrame(rows), False, 64, 42, mode='val', forensic_mode='jpeg')
    samples = [dataset[i] for i in range(2)]
    batch = BatchTransfer('cpu')(ValidationCollator()(samples))
    assert 'fmap' not in batch
    assert [x['bins'].shape for x in batch['jpeg']] == [(40, 56), (40, 64)]
    native = JPEGInput.read(jpeg_bytes(value=200))
    image = np.full((19, 27, 3), 200, np.uint8)
    sample = DataSample(image=image, jpeg=native)
    result = RandomJPEGRecompression((60, 61), 1).apply(sample, np.random.default_rng(1))
    assert not np.array_equal(result.jpeg.qtable, native.qtable)
    np.testing.assert_array_equal(result.qtable, result.jpeg.qtable)


def test_jpeg_model_forward_backward_and_required_inputs():
    from src.config import ModelConfig
    from src.training.builders import build_model
    torch.set_num_threads(1)
    model = build_model(ModelConfig(forensic_mode='jpeg', dct_aux_weight=0.1), pretrained=False).train()
    inputs = [JPEGInput.read(jpeg_bytes((h, w), value=200)).tensors() for h, w in ((64, 96), (80, 64))]
    image = torch.randn(2, 3, 64, 64)
    output = model(image, jpeg=inputs)
    assert output['logits'].shape == (2, 1, 64, 64)
    output['dct_aux_logits'].square().mean().backward()
    assert model.forensic_fusion.branch.artifact.dc_layer0_dil[0].weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match='jpeg'):
        model(image)


def test_feature_geometry_matches_pixel_operations():
    from src.modules.jpeg_branch import JPEGBranch
    feature = torch.arange(24.).reshape(1, 1, 4, 6)
    actual = JPEGBranch.align(feature, (8, 16, 16, 24, 1, 1, 0), 8, (3, 2))
    expected = feature[:, :, 1:3, 2:5].rot90(1, (-2, -1)).flip(-1)
    torch.testing.assert_close(actual, expected)


def test_three_ablation_recipes_and_explicit_native_budget():
    from src.budget import count_gflops
    from src.config import load_experiment_config
    from src.training.builders import build_model
    configs = [load_experiment_config(f'configs/{name}.yaml') for name in ('dct576', 'rgb576', 'jpeg576')]
    assert all(c.dataset.image_size == 576 for c in configs)
    assert configs[0].train == configs[1].train == configs[2].train
    assert configs[0].loss == configs[1].loss == configs[2].loss
    assert configs[0].augmentation == configs[1].augmentation == configs[2].augmentation
    assert [c.model.use_forensics for c in configs] == [True, False, True]
    with torch.device('meta'):
        model = build_model(configs[2].model, pretrained=False)
        with pytest.raises(ValueError, match='native_size'):
            count_gflops(model, 576)
        assert count_gflops(model, 576, native_size=(1024, 1024)) < 100
        assert count_gflops(model, 576, native_size=(1080, 1920)) > 100


@pytest.mark.parametrize('orientation', range(1, 9))
def test_exif_alignment_matches_rgb_orientation(orientation):
    from PIL import ImageOps

    from src.modules.jpeg_branch import JPEGBranch
    raw = np.arange(24, dtype=np.uint8).reshape(4, 6)
    image = Image.fromarray(raw)
    image.getexif()[274] = orientation
    expected = np.array(ImageOps.exif_transpose(image))
    feature = torch.tensor(raw, dtype=torch.float32)[None, None]
    actual = JPEGBranch.align(feature, (0, 0, *expected.shape, 0, 0, 0), 1, expected.shape,
                              orientation=orientation, source_size=raw.shape)
    torch.testing.assert_close(actual[0, 0], torch.tensor(expected, dtype=torch.float32))
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', exif=image.getexif())
    native = JPEGInput.read(buffer.getvalue())
    assert native.geometry[2:4] == expected.shape
