import numpy as np
import pytest

from src.forensic.dct import forensic_maps


@pytest.mark.parametrize("height,width", [(8, 8), (24, 40), (27, 43)])
def test_dct_maps_keep_block_grid_for_non_square_and_unaligned_images(height, width):
    image = np.random.default_rng(42).integers(0, 256, (height, width, 3), dtype=np.uint8)
    maps = forensic_maps(image)
    assert maps.shape == (12, height // 8, width // 8)
    assert maps.dtype == np.float32
    assert np.isfinite(maps).all()
