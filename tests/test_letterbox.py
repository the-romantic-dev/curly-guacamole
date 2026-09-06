from dataclasses import replace

import numpy as np
import pytest
import torch

from src.data.data_sample import DataSample
from src.data.preprocess import SamplePreprocessor
from src.losses import compute_loss
from src.modules.gate_head import GateHead


@pytest.mark.parametrize("shape,content", [((24, 40), (38, 64)), ((40, 24), (64, 38)), ((32, 32), (64, 64))])
def test_letterbox_preserves_ratio_and_marks_padding(shape, content):
    sample = DataSample(np.full((*shape, 3), 100, np.uint8),
                        np.ones(shape, np.float32), np.ones((12, shape[0]//8, shape[1]//8), np.float32))
    processor = SamplePreprocessor(64, resize_mode="letterbox")
    output = processor.to_output(processor.resize(sample))
    h, w = content
    assert output["content_size"].tolist() == [h, w]
    assert output["image"].shape == (3, 64, 64)
    assert output["valid_mask"].sum() == h*w
    assert output["mask"].sum() == h*w
    assert output["fmap"].shape == (12, 8, 8)
    assert torch.count_nonzero(output["image"][:, h:]) == 0
    assert torch.count_nonzero(output["image"][:, :, w:]) == 0


def test_padding_does_not_change_main_or_aux_losses():
    logits = torch.randn(2, 1, 8, 8, requires_grad=True)
    valid = torch.zeros_like(logits)
    valid[:, :, :4, :6] = 1
    batch = {"mask": torch.ones_like(logits), "label": torch.ones(2,1), "valid_mask": valid}
    out = {"logits": logits, "aux_logits": logits, "cls_logits": torch.zeros(2,1)}
    loss = compute_loss(out, batch, .4)
    other = logits.detach().clone()
    other[valid == 0] = 100
    changed = compute_loss({**out, "logits": other, "aux_logits": other}, batch, .4)
    assert loss.item() == pytest.approx(changed.item())
    loss.backward()
    assert not logits.grad[valid == 0].any()


def test_dct_coordinates_and_partial_block_coverage():
    from src.data.letterbox import Letterbox

    # 24x40 -> 38x64: block centers must retain their image coordinates.
    yy, xx = np.meshgrid(np.arange(3), np.arange(5), indexing="ij")
    maps = np.stack([xx, yy] * 6).astype(np.float32)
    sample = DataSample(np.zeros((24,40,3), np.uint8), fmap=maps)
    output = Letterbox(64).apply(sample)
    # Destination block (2,3): source x=3.5*40/64-.5=1.6875,
    # y=2.5*24/38-.5=1.07895, quantized by OpenCV to 1/32 pixel.
    assert output.fmap[0,2,3] == pytest.approx(1.6875)
    assert output.fmap[1,2,3] == pytest.approx(1.07895, abs=1/32)
    assert output.fmap[0,4,3] == pytest.approx(1.6875 * .75)
    assert not output.fmap[:,5:].any()


def test_classification_pool_ignores_padding():
    head = GateHead(1, dropout=0).eval()
    with torch.no_grad():
        head.fc.weight.fill_(1)
        head.fc.bias.zero_()
    features = torch.tensor([[[[2., 100.], [2., 100.]]]])
    valid = torch.zeros(1, 1, 8, 8)
    valid[:, :, :, :4] = 1
    assert head(features, valid_mask=valid).item() == pytest.approx(2.)


def test_unpadding_precedes_restore_and_area_gate():
    from src.inference.predict import Predictor, ThresholdConfig
    from src.training.builders import AmpContext
    predictor = Predictor(torch.nn.Identity(), ThresholdConfig(.5, 0, .75),
                          AmpContext(torch.device("cpu"), torch.float32, False, False))
    probability = torch.zeros(1, 1, 8, 8)
    probability[:, :, :4] = 1
    result = predictor.binary_mask(probability, 1., (12, 24), content_size=(4, 8))
    assert result.shape == (12, 24)
    assert np.all(result == 255)


def test_letterbox_experiment_is_separate_and_serialized():
    from src.config import ExperimentConfig, load_experiment_config
    from src.inference.submission import InferenceConfig
    base = load_experiment_config("configs/efficientvit_b2_mixed_original.yaml")
    config = load_experiment_config("configs/efficientvit_b2_letterbox.yaml")
    assert config.dataset == replace(base.dataset, resize_mode="letterbox")
    assert config.paths.run_name != base.paths.run_name
    assert config.train == base.train
    assert config.eval == base.eval
    assert ExperimentConfig.from_dict(config.to_dict()) == config
    assert InferenceConfig.from_snapshot(config.to_flat_dict()).resize_mode == "letterbox"


@pytest.mark.parametrize("resolution", ["original", "resized"])
def test_validation_and_predictor_ignore_padding_end_to_end(resolution):
    from src.config import load_experiment_config
    from src.inference.predict import Predictor, ThresholdConfig
    from src.training.builders import build_amp
    from src.training.validation import validate

    class MaskModel(torch.nn.Module):
        def forward(self, image, fmap=None, valid_mask=None):
            assert valid_mask is not None
            return {"logits": image[:, :1], "cls_logits": torch.full((len(image), 1), 10.)}

    config = load_experiment_config("configs/efficientvit_b2_letterbox.yaml")
    config = replace(config, train=replace(config.train, device="cpu", amp="off"),
                     eval=replace(config.eval, resolution=resolution, mask_thresholds=(.5,),
                                  cls_thresholds=(0.,), min_areas=(0.,)))
    valid = torch.zeros(2, 1, 8, 8, dtype=torch.bool)
    valid[0, :, :4, :] = True
    valid[1, :, :, :4] = True
    image = torch.full((2, 3, 8, 8), 10.)  # Intentionally strong false predictions on padding.
    image[1, :, :, :4] = -10.
    masks = valid.float()
    masks[1] = 0
    batch = {"image": image, "fmap": torch.zeros(2, 12, 1, 1), "mask": masks,
             "valid_mask": valid, "content_size": torch.tensor([[4,8],[8,4]]),
             "original_mask": [torch.ones(6,12,dtype=torch.bool), torch.zeros(12,6,dtype=torch.bool)],
             "original_size": torch.tensor([[6,12],[12,6]]), "image_path": ["p.jpg", "n.jpg"]}
    amp = build_amp(config.train)
    result = validate(MaskModel(), [batch], amp, config, torch.device("cpu"))
    assert result.tuned.aic > .999
    assert result.tuned.fpr_neg == 0
    assert result.accumulator.n_pixels == ([72,72] if resolution == "original" else [32,32])
    predictions = list(Predictor(MaskModel(), ThresholdConfig(), amp).predict([batch]))
    assert np.all(predictions[0].mask == 255)
    assert not predictions[1].mask.any()


def test_full_valid_mask_matches_legacy_loss():
    logits = torch.randn(2,1,4,4)
    out = {"logits": logits, "aux_logits": logits, "cls_logits": torch.zeros(2,1)}
    batch = {"mask": torch.ones_like(logits), "label": torch.ones(2,1)}
    legacy = compute_loss(out, batch, .4)
    padded = compute_loss(out, {**batch, "valid_mask": torch.ones_like(logits)}, .4)
    assert legacy.item() == pytest.approx(padded.item())


def test_letterbox_flops_include_masked_classification():
    from torch._subclasses.fake_tensor import FakeTensorMode

    from src.budget import count_gflops
    from src.config import load_experiment_config
    from src.training.builders import build_model

    config = load_experiment_config("configs/efficientvit_b2_letterbox.yaml")
    with FakeTensorMode():
        model = build_model(config.model, pretrained=False).eval()
        flops = count_gflops(model, config.dataset.image_size, use_valid_mask=True)
    assert flops < 100
