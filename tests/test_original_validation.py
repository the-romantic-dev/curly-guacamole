import numpy as np
import pandas as pd
import pytest
from PIL import Image


def test_original_index_deduplicates_pixels_and_rejects_wrong_pair(tmp_path):
    from src.eval.originals import OriginalIndex

    rng = np.random.default_rng(2)
    image = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    Image.fromarray(image).save(tmp_path / 'a.png')
    Image.fromarray(image).save(tmp_path / 'b.png')
    changed = image.copy()
    changed[20:40, 20:40] = 255
    Image.fromarray(changed).save(tmp_path / 'changed.png')
    Image.fromarray(np.full_like(image, 0)).save(tmp_path / 'wrong.png')
    mask = np.zeros((64, 64), np.uint8)
    mask[20:40, 20:40] = 255
    Image.fromarray(mask).save(tmp_path / 'mask.png')
    rows = pd.DataFrame([
        dict(orgl_img_path=p, chng_img_path='changed.png', gt_path='mask.png', group_id=g, domain='coco')
        for p, g in [('a.png', 'a'), ('b.png', 'b'), ('wrong.png', 'c'), ('missing.png', 'd')]
    ])
    index = OriginalIndex(tmp_path)
    originals, pairs = index.build(rows, workers=1)
    assert len(originals) == 1
    assert set(originals.iloc[0].group_ids) == {'a', 'b'}
    assert pairs.status.tolist() == ['accepted', 'accepted', 'unmatched', 'unreadable_original']
    assert originals.iloc[0].target_kind == 'original_zero'


def test_explicit_original_gets_zero_mask_but_missing_gt_is_still_error(tmp_path):
    from src.data.sample_io import SampleIO

    io = SampleIO(tmp_path, True)
    original = pd.DataFrame([dict(chng_img_path='original.jpg', gt_path=None, target_kind='original_zero')])
    io.validate_dataframe(original)
    np.testing.assert_array_equal(io.load_mask(original.iloc[0], (8, 10)), np.zeros((8, 10)))
    with pytest.raises(ValueError):
        io.validate_dataframe(original.drop(columns='target_kind'))


def test_original_index_does_not_read_outside_dataset(tmp_path):
    from src.eval.originals import OriginalIndex

    rows = pd.DataFrame([dict(orgl_img_path='../elsewhere.jpg', chng_img_path='x.jpg',
                              gt_path='x.png', group_id='x', domain='coco')])
    originals, pairs = OriginalIndex(tmp_path).build(rows, workers=1)
    assert originals.empty
    assert pairs.iloc[0].status == 'unreadable_original'
