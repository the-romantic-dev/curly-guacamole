# AIIJC segmentation experiments

Reusable code lives in `src/`, experiment settings in `configs/`, and results in
`runs/<run_name>/`. Use the `challenges` conda environment.

## Training

Open `notebooks/baseline_pipeline.ipynb` with the `challenges` kernel. For a new
experiment, create a YAML inheriting from an existing config, set `run_name`, and
select the new YAML in the notebook. The same pipeline can run from a Python script:

```python
from src.config import load_experiment_config
from src.training.engine import run_experiment

run = run_experiment(load_experiment_config("configs/baseline.yaml"))
print(run.summary)
```

For example, `configs/my_experiment.yaml` can contain only the overrides:

```yaml
extends: baseline_mixed_original.yaml
paths:
  run_name: my_experiment
dataset:
  image_size: 672
train:
  batch_size: 2
  accum_steps: 8
```

`extends` names one parent YAML, relative to the file containing it (absolute
paths also work). Parents can inherit from other configs. Nested mappings merge
recursively; child values win, and lists replace the entire inherited list.
Omitted fields are inherited; an empty mapping does not clear inherited keys.
Cycles, missing parents, and invalid settings raise errors during loading.
Changing a parent affects its descendants on their next load. Run snapshots
contain the fully resolved settings, without `extends`, and remain self-contained.

Inspect `summary.json`, `metrics.csv`, and `notes.md` in the run directory.
`notebooks/16_forensic_maps_before_resize.ipynb` is a historical research archive;
its old API is not the supported training entry point.

## Mixed-frame experiment

Open `notebooks/baseline_mixed_original.ipynb` (restart the kernel after updating
the package). It uses `configs/baseline_mixed_original.yaml` and writes to
`runs/pvt_v2_b2_mixed_original`, keeping the completed baseline separate.
The architecture, fold, seed and training budget stay at 8 × 24,000 = 192,000
sample presentations.

- `full_frame_probability: 0.5`: first six epochs mix full frames and square crops.
- `foreground_crop_probability: 0.5`: half of crop attempts select a GT pixel
  and an 8-aligned window containing it; empty masks use ordinary random crops.
- `final_full_frame_epochs: 2`: last two epochs use full frames. Full-frame
  examples keep flips but do not rotate by 90 degrees.
- `full_frame: true` still overrides mixing and always selects a full frame.
- `eval.resolution: original`: restore sigmoid probabilities with bilinear
  interpolation before histogram threshold tuning against the original binary GT.
  Different mask sizes are retained in a list by `ValidationCollator`.
  Malformed GT with a different size from its image raises an error.

RNG streams advance on repeated samples and reset reproducibly per epoch and
worker; the epoch is shared with persistent workers. Sampling is seeded per
epoch as well. Reproducibility assumes the same worker count and batch setup;
the existing checkpoint format does not guarantee bit-identical training resume
for model-side randomness such as dropout.
`train/negative_fraction` records the observed fraction after cropping and resize,
which can differ from the sampler's configured negative fraction.

Legacy YAML files default to crop-only augmentation and resized validation.
The RNG fix applies to them too. Do not compare the completed baseline's resized
AIC directly with the new original-resolution AIC: re-evaluate its saved best
checkpoint on the same validation rows at original resolution first. Subsequent
fold checks should use a new run name and the same number of sample presentations.
Changing the new augmentation options or validation resolution when resuming an
existing run is rejected before its snapshot can be overwritten.

## EfficientViT experiment

`notebooks/efficientvit_b2_mixed_original.ipynb` uses
`configs/efficientvit_b2_mixed_original.yaml`. The existing `Segmenter` accepts
the MIT EfficientViT encoder through timm; no encoder adapter is required.
The backbone is `efficientvit_b2.r288_in1k` (ImageNet classification weights),
from [MIT EfficientViT](https://github.com/mit-han-lab/efficientvit), corresponding
to [the multi-scale linear attention paper](https://arxiv.org/abs/2205.14756).
This is the timm `efficientvit_b*` family, distinct from `efficientvit_m*`.

Its features have strides `[4, 8, 16, 32]` and channels `[48, 96, 192, 384]`.
The same DCT branch and gated residual fusion attach at strides 8/16/32;
decoder channels, auxiliary supervision and classification head are retained.
At **1024 × 1024**, the complete model has **16,310,395 parameters** and costs
**97.00815744 GFLOPS** per image (batch 1, eval, `FlopCounterMode`, confirmed by
a real CPU forward with random weights). The PVT baseline at 640 costs 97.1621.
The CPU DCT preprocessing, mask restoration and metric calculation are outside
this forward count. H100 latency and full-resolution training GPU memory are
not yet measured; the active GPU training run was not interrupted.

The experiment keeps the same fold, seed, augmentation schedule, original-size
validation and 192,000 presentations. Batch 2 with accumulation 8 preserves an
effective batch of 16; BatchNorm still observes the physical batch of 2.
Results go to `runs/efficientvit_b2_1024_mixed_original`.
First training use downloads the configured pretrained weights through timm;
the compatibility checks use `pretrained=False` and require no download.
If GPU memory is insufficient, use batch 1 / accumulation 16 in a copied config
with a new run name. Start this notebook after the current GPU experiment ends.

For reference, measured full-model alternatives with the same decoder:

| Encoder | Image size | GFLOPS |
| --- | ---: | ---: |
| EfficientViT-B1 | 1024 | 49.982 |
| EfficientViT-B2 | 896 | 74.272 |
| EfficientViT-B2 | 1024 | 97.008 |
| EfficientViT-B3 | 768 | 112.100 (over budget) |

## EfficientViT with aspect-preserving resize

`notebooks/efficientvit_b2_letterbox.ipynb` selects
`configs/efficientvit_b2_letterbox.yaml` and writes to
`runs/efficientvit_b2_1024_letterbox`. All training settings match the EfficientViT
mixed/original experiment; only `dataset.resize_mode: letterbox` changes geometry
and enables padding-aware loss, classification pooling and restoration.

The longer side is resized to 1024 (including upscaling smaller images), the
shorter side is rounded to the nearest pixel, and padding is added on the bottom
and right. For example, 640×480 becomes 1024×768 plus 256 bottom rows. The top-left
origin remains aligned with the DCT grid. Square random training crops remain
square; full-frame examples retain their aspect ratio.

DCT maps are extracted before resizing. Their block centers are resampled with
the same image scale, and partial boundary blocks are weighted by valid coverage.
RGB padding is reset to zero after normalization and photometric augmentation;
GT and forensic padding are zero. `valid_mask` excludes padding from BCE/Dice
(including auxiliary loss) and weights classification pooling at encoder resolution.
Convolutions, attention and normalization still operate on the full square.

Validation and submission share the same inverse: crop to `content_size`, resize
probabilities to the original dimensions, then binarize. Padding never enters the
metric area denominator, including the optional resized validation mode. The resize
mode is saved in the snapshot and automatically restored by submission inference.
Legacy configs/checkpoints default to `stretch`; changing geometry on resume is rejected.

The model still processes a dense 1024×1024 tensor, so padding does not save FLOPs.
The counted forward includes the masked classification path; preprocessing,
restoration, and validation histograms remain outside that count.

## Submission

From the project root in PowerShell:

```powershell
& 'D:/Apps/anaconda3/envs/challenges/python.exe' -m src.inference runs/<run_name> submissions/<run_name>
```

This reads `ckpt/best.pt` (EMA weights when present), the saved configuration, and
the tuned thresholds in `summary.json`. It writes `submission.csv` and the PNG
masks under `predictions/`, retaining template paths and original image sizes.
For the competition archive, zip those two entries at the archive root.

Console output reports configuration/data/checkpoint loading, the selected device
and thresholds, prediction and PNG saving progress by batch, and the final output
directory. Progress is printed after the first batch, every 30 seconds, and after
the last batch, with elapsed time and an estimated remaining time once available.

Use `--data-path` to relocate the dataset, `--template-path` for an explicit
template, or `--device cpu` to override the saved device. To override the operating
point, pass all three flags: `--mask-threshold`, `--cls-threshold`, `--min-area`.
Histogram tuning reports the effective bin boundary used during validation.
Legacy validation scores use resized targets; final predictions are resized to
original dimensions before binarization. Set `eval.resolution: original` for the
same restoration and thresholding order during model selection.

## Experiment history

Open `notebooks/experiments.ipynb` to view the experiment tree and results table.
Edit the YAML header in `runs/<run_name>/notes.md` by hand:

```yaml
parent: pvt_v2_b2_mixed_original
baseline: null
change: "Replaced the decoder with SegFormer"
leaderboard_score: null
```

`parent` is the experiment this run was derived from, not an instruction to load
weights. A missing, empty or null `baseline` defaults to `parent`; set another
single run name to override it. References use run directory names. Root runs
leave `parent` empty. Existing baseline-only cards keep their meaning.
Replace `leaderboard_score: null` with a number from 0 to 1 after scoring.
No private score or submission metadata is required.

Cards are tracked by Git; training artifacts remain ignored. Existing cards are
never overwritten by training. New runs receive the expanded template automatically.
Local metrics still come from `summary.json`. The registry reports missing links,
cycles and malformed records without changing run files. Recreate the registry
or rerun the notebook after edits. Runs without cards remain visible.

```python
from src.training.experiments import ExperimentRegistry

registry = ExperimentRegistry("runs")
print(registry.tree())
print(registry.table())
print(registry.issues)
```

Compare validation scores only on matching validation rows and evaluation
protocols; in particular, resized and original-resolution AIC are not directly
comparable. The overview does not calculate automatic score differences.

## Regression checks

```powershell
& 'D:/Apps/anaconda3/envs/challenges/python.exe' -m pytest
```

The suite covers configuration, factories, training state, validation, dataset
modes and paths, DCT shapes, synchronized augmentation, model forward, and
template-aligned submission. Tests use synthetic inputs and mocked encoders, with
no training dataset or pretrained weight download required.
## Storage paths

Experiment YAML files only specify `paths.run_name`. Copy `.env.example` to
`.env` in the project root and set `AIIJC_DATA_PATH` and `AIIJC_RUNS_PATH` once
per machine. `.env` is ignored by Git; `.env.example` is the shared template.
The file is read each time an experiment config is loaded, including in notebooks.
Environment variables override `.env`; missing or empty values default to
the project's `data/` and `runs/` directories.
Relative storage paths resolve from the project root, regardless of the
working directory or YAML location.

For example, a server's `.env`:

```dotenv
AIIJC_DATA_PATH=/mnt/datasets/aiijc
AIIJC_RUNS_PATH=/mnt/experiments/aiijc
```

The same YAML and notebook work on both machines. Runtime snapshots retain
the resolved paths for reference; loading an old experiment YAML ignores its
stored `data_path` and `runs_path` in favor of the current machine settings.
