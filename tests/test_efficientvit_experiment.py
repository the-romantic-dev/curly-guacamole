from dataclasses import replace

import pytest
import torch

from src.budget import count_gflops
from src.config import load_experiment_config
from src.losses import compute_loss
from src.training.builders import build_model


@pytest.fixture
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_efficientvit_config_preserves_mixed_protocol():
    baseline = load_experiment_config("configs/baseline_mixed_original.yaml")
    config = load_experiment_config("configs/efficientvit_b2_mixed_original.yaml")
    assert config.paths.run_name != baseline.paths.run_name
    assert config.augmentation == baseline.augmentation
    assert config.eval == baseline.eval
    assert config.dataset == replace(baseline.dataset, image_size=1024)
    assert config.model == replace(baseline.model, encoder_name="efficientvit_b2.r288_in1k")
    assert config.train.epochs * config.train.epoch_size == 192000
    assert config.train.batch_size * config.train.accum_steps == 16


def test_efficientvit_real_encoder_trains_with_forensic_fusion(cpu_threads):
    config = load_experiment_config("configs/efficientvit_b2_mixed_original.yaml")
    model = build_model(config.model, pretrained=False).train()
    assert model.strides == [4, 8, 16, 32]
    assert model.channels == [48, 96, 192, 384]
    batch = {
        "image": torch.randn(2, 3, 64, 64),
        "fmap": torch.randn(2, 12, 8, 8),
        "mask": torch.ones(2, 1, 64, 64),
        "label": torch.ones(2, 1),
    }
    # Nonzero fusion gates model the state after they start learning.
    with torch.no_grad():
        for block in model.forensic_fusion.fusion_blocks.values():
            block.channel_gate.fill_(0.1)
    output = model(batch["image"], batch["fmap"])
    assert output["logits"].shape == (2, 1, 64, 64)
    assert output["aux_logits"].shape == output["logits"].shape
    assert output["cls_logits"].shape == (2, 1)
    loss = compute_loss(output, batch, config.model.aux_weight)
    loss.backward()
    assert torch.isfinite(loss)
    for module in (model.encoder, model.forensic_fusion.branch, model.decoder):
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(grad).all() for grad in grads)
        assert any(grad.abs().sum() > 0 for grad in grads)
    model.eval()
    with torch.inference_mode():
        actual = model(batch["image"], batch["fmap"])["logits"]
        no_forensics = model(batch["image"], torch.zeros_like(batch["fmap"]))["logits"]
    assert not torch.equal(actual, no_forensics)


def test_efficientvit_full_model_fits_1024_flop_budget(cpu_threads):
    from torch._subclasses.fake_tensor import FakeTensorMode

    config = load_experiment_config("configs/efficientvit_b2_mixed_original.yaml")
    # Fake CPU tensors support EfficientViT's device-specific autocast context;
    # meta tensors do not. No weights or GPU memory are allocated.
    with FakeTensorMode():
        model = build_model(config.model, pretrained=False).eval()
        gflops = count_gflops(model, config.dataset.image_size)
    assert gflops == pytest.approx(97.00815744)
    assert gflops <= 100
