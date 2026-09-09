from dataclasses import replace

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from src.config import load_experiment_config
from src.data.data_workspace import DataWorkspace
from src.training.builders import build_datasets


@pytest.mark.parametrize('device,amp,dtype', [('cuda', 'bf16', torch.bfloat16),
                                           ('cuda', 'fp16', torch.float16),
                                           ('cuda', 'off', torch.float32),
                                           ('cpu', 'bf16', torch.float32)])
def test_local_transport_matches_amp_cast(tmp_path, device, amp, dtype):
    cfg = load_experiment_config('configs/local.yaml')
    cfg = replace(cfg, model=replace(cfg.model, local_image_size=64),
                  dataset=replace(cfg.dataset, image_size=32),
                  train=replace(cfg.train, device=device, amp=amp))
    workspace = DataWorkspace(tmp_path)
    workspace.train_root.mkdir(parents=True)
    image = np.random.default_rng(1).integers(0, 256, (83, 111, 3), dtype=np.uint8)
    cv2.imwrite(str(workspace.train_root / 'a.png'), image)
    cv2.imwrite(str(workspace.train_root / 'mask.png'), np.zeros((83, 111), np.uint8))
    rows = pd.DataFrame({'chng_img_path': ['a.png'], 'gt_path': ['mask.png'], 'is_negative': [True]})
    train, val = build_datasets(cfg, workspace, rows, rows)
    train.use_augmentations = False
    sample = train[0]
    from src.data.local_preprocess import LocalPreprocessor
    expected = LocalPreprocessor(64)(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).to(dtype)
    assert sample['local_input'].dtype == dtype
    torch.testing.assert_close(sample['local_input'], expected, rtol=0, atol=0)
    assert val[0]['local_input'].dtype == dtype
    assert sample['mask'].dtype == torch.float32


def test_batch_transfer_cpu_preserves_non_tensor_metadata():
    from src.training.transfer import BatchTransfer
    batch = {'image': torch.randn(2, 3, 8, 8), 'image_path': ['a', 'b']}
    actual = BatchTransfer('cpu')(batch)
    assert actual['image'] is batch['image']
    assert actual['image_path'] is batch['image_path']


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA stream integration requires GPU')
def test_cuda_copy_stream_values_and_lifetime():
    from src.training.transfer import BatchTransfer
    transfer = BatchTransfer('cuda')
    outputs = []
    for i in range(20):
        batch = {'local_input': torch.full((2, 15, 64, 64), float(i), dtype=torch.float16).pin_memory()}
        moved = transfer(batch)
        outputs.append(moved['local_input'].float().sum())
        del batch, moved
    torch.cuda.synchronize()
    assert [value.item() for value in outputs] == [float(i * 2 * 15 * 64 * 64) for i in range(20)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Requires CUDA autocast')
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_compact_local_branch_matches_cuda_autocast_and_gradients(dtype):
    from src.modules.local_branch import LocalBranch
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip('BF16 not supported on this GPU')
    torch.manual_seed(8)
    model = LocalBranch(16, True).cuda().train()
    native = torch.randn(2, 15, 64, 64)
    global_features = torch.randn(2, 16, 8, 8, device='cuda')
    torch.testing.assert_close(native.to(dtype).cuda(), native.cuda().to(dtype), rtol=0, atol=0)
    results, gradients = [], []
    for local in (native.cuda(), native.to(dtype).cuda()):
        model.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=dtype):
            out, aux = model(local, global_features)
            loss = out.float().square().mean() + aux.float().square().mean()
        loss.backward()
        results.append((out.detach(), aux.detach()))
        gradients.append([p.grad.clone() for p in model.parameters()])
    for expected, actual in zip(results[0], results[1]):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for expected, actual in zip(gradients[0], gradients[1]):
        # CUDA backward reductions may differ in their last float32 bits.
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-8)
