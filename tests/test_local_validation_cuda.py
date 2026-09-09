"""Full-resolution validation smoke test for the server's local/BF16 recipe."""

from dataclasses import replace

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from src.config import load_experiment_config
from src.data.data_workspace import DataWorkspace
from src.training.builders import build_amp, build_datasets, build_ema, build_loaders, build_model
from src.training.validation import validate


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Requires CUDA')
def test_local_validation_bf16_full_resolution_and_partial_batch(tmp_path):
    if not torch.cuda.is_bf16_supported():
        pytest.skip('Requires BF16 support')
    cfg = load_experiment_config('configs/local.yaml')
    cfg = replace(cfg, train=replace(cfg.train, device='cuda', amp='bf16', workers=4, batch_size=8))
    assert cfg.dataset.image_size == 640 and cfg.model.local_image_size == 1024
    workspace = DataWorkspace(tmp_path)
    workspace.train_root.mkdir(parents=True)
    rows = []
    shapes = [(37, 65), (103, 79), (720, 1280), (1080, 1920), (96, 64),
              (80, 73), (192, 256), (48, 80), (99, 151)]
    rng = np.random.default_rng(23)
    for index, shape in enumerate(shapes):
        image = rng.integers(0, 256, (*shape, 3), dtype=np.uint8)
        cv2.imwrite(str(workspace.train_root / f'{index}.png'), image)
        original = index % 2 == 1 or index == 8
        positive = index in (0, 2)
        mask_path = None
        if not original:
            # Include supplied GT smaller than the corresponding image.
            mask_shape = (360, 640) if index == 2 else shape
            mask = np.zeros(mask_shape, dtype=np.uint8)
            if positive:
                mask[2:8, 3:11] = 255
            mask_path = f'{index}_mask.png'
            cv2.imwrite(str(workspace.train_root / mask_path), mask)
        rows.append(dict(chng_img_path=f'{index}.png', gt_path=mask_path,
                         target_kind='original_zero' if original else 'provided',
                         is_negative=not positive))
    rows = pd.DataFrame(rows)
    datasets = build_datasets(cfg, workspace, rows, rows)
    _, loader = build_loaders(cfg.train, *datasets)
    model = build_model(cfg.model, pretrained=False).to('cuda', memory_format=torch.channels_last)
    ema = build_ema(cfg.train, model)
    seen = []

    def inspect_batch(module, args, kwargs):
        seen.append(len(args[0]))
        assert kwargs['local_input'].dtype == torch.bfloat16
        assert not module.training

    handle = ema.module.register_forward_pre_hook(inspect_batch, with_kwargs=True)
    torch.cuda.reset_peak_memory_stats()
    result = validate(ema.module, loader, build_amp(cfg.train), cfg, torch.device('cuda'))
    handle.remove()
    assert seen == [8, 1]
    assert result.resolution == 'original'
    assert result.tuned.n_pos == 2 and result.tuned.n_neg == 7
    assert result.fixed is not None
    assert all(np.isfinite(value) for value in result.loss_components.values())
    assert all(np.isfinite(value) for value in result.tuned.as_dict().values())
    print(f'Full local validation peak allocated: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB')
