import numpy as np
import pytest

from src.data.augmentation.transforms.random_dct_aligned_crop import RandomDCTAlignedCrop
from src.data.augmentation.transforms.random_rotate_flip import RandomRotateFlip
from src.data.data_sample import DataSample


@pytest.mark.parametrize("seed", range(10))
def test_crop_and_flip_keep_image_mask_and_dct_blocks_aligned(seed):
    blocks = np.arange(16 * 20, dtype=np.float32).reshape(16, 20)
    pixels = blocks.repeat(8, axis=0).repeat(8, axis=1)
    sample = DataSample(
        image=np.repeat(pixels[..., None], 3, axis=2),
        mask=pixels.copy(),
        fmap=np.repeat(blocks[None], 12, axis=0),
    )
    rng = np.random.default_rng(seed)
    cropped = RandomDCTAlignedCrop((0.25, 0.5), full_frame=False).apply(sample, rng)
    transformed = RandomRotateFlip(full_frame=False).apply(cropped, rng)

    assert cropped.image.shape[0] < sample.image.shape[0]
    np.testing.assert_array_equal(transformed.image[..., 0], transformed.mask)
    for channel in transformed.fmap:
        np.testing.assert_array_equal(channel.repeat(8, axis=0).repeat(8, axis=1), transformed.mask)
    assert transformed.image.flags.c_contiguous
    assert transformed.fmap.flags.c_contiguous
