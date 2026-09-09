# EMCAD decoder

`Segmenter` defaults to PVTv2-B2 with `EMCADDecoder`, the existing forensic
fusion branch, and an optional local branch. Construct EMCAD directly:

```python
from src.decoders import EMCADDecoder

decoder = EMCADDecoder(
    encoder_channels=[64, 128, 320, 512],
    encoder_strides=[4, 8, 16, 32],
    norm="batch",
    use_aux=True,
)
features, aux_logits = decoder(encoder_features)
```

The four input features run from fine to coarse. Channels must be positive
and even. Output features have stride 4 and the finest encoder width;
explicit skip sizes support odd and rectangular geometry. Auxiliary logits
are returned only during training when enabled.

This adaptation uses channel attention, shared spatial attention, parallel
additive depthwise convolutions, channel shuffle, residual connections,
efficient upsampling, and grouped skip attention. Auxiliary supervision
uses the merged final skip before refinement. The paper's multiple mask
heads and training scheme are not reproduced.

`model.decoder_kwargs` passes EMCAD options directly: `kernel_sizes`,
`expansion_factor`, `lgag_kernel_size`, `activation`, and optional refinement
widths. Unknown options raise TypeError. Segmenter supplies encoder metadata,
`norm`, and `use_aux`; these cannot be overridden through decoder_kwargs.
There is no decoder registry or decoder selection argument.

`configs/baseline.yaml` uses the default decoder. `configs/stride4.yaml`
sets `output_refinement_channels: 144`; `configs/stride2_rgb.yaml` sets
`rgb_refinement_channels: 32` and `rgb_detail_channels: 24`. The latter
combines RGB, decoder features, and coarse logits to refine logits at
stride 2. `output_stride` still describes the stride-4 decoder features.
The two refinements are mutually exclusive, and neither can be combined
with `local_image_size` in `configs/local.yaml`.

Segmenter owns the mask and classification heads and restores logits to
input size. Module names (`encoder`, `decoder`, `forensic_fusion`,
`segmentation_head`, `classification_head`, `local_branch`) retain the
existing EMCAD checkpoint layout. Removed UNet/SegFormer snapshots are
not supported by this API.

Architecture reference: [EMCAD, CVPR 2024](https://arxiv.org/abs/2405.06880).
