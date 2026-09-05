# AIIJC segmentation experiments

Reusable code lives in `src/`, experiment settings in `configs/`, and results in
`runs/<run_name>/`. Use the `challenges` conda environment.

## Training

Open `notebooks/baseline_pipeline.ipynb` with the `challenges` kernel. For a new
experiment, copy `configs/baseline.yaml`, change its parameters and `run_name`, and
select the new YAML in the notebook. The same pipeline can run from a Python script:

```python
from src.config import load_experiment_config
from src.training.engine import run_experiment

run = run_experiment(load_experiment_config("configs/baseline.yaml"))
print(run.summary)
```

Inspect `summary.json`, `metrics.csv`, and `notes.md` in the run directory.
`notebooks/16_forensic_maps_before_resize.ipynb` is a historical research archive;
its old API is not the supported training entry point.

## Submission

From the project root in PowerShell:

```powershell
& 'D:/Apps/anaconda3/envs/challenges/python.exe' -m src.inference runs/<run_name> submissions/<run_name>
```

This reads `ckpt/best.pt` (EMA weights when present), the saved configuration, and
the tuned thresholds in `summary.json`. It writes `submission.csv` and the PNG
masks under `predictions/`, retaining template paths and original image sizes.
For the competition archive, zip those two entries at the archive root.

Use `--data-path` to relocate the dataset, `--template-path` for an explicit
template, or `--device cpu` to override the saved device. To override the operating
point, pass all three flags: `--mask-threshold`, `--cls-threshold`, `--min-area`.
Histogram tuning reports the effective bin boundary used during validation.
Validation scores use resized targets; final predictions are resized to original
dimensions before binarization, so their scores need not be identical.

## Regression checks

```powershell
& 'D:/Apps/anaconda3/envs/challenges/python.exe' -m pytest
```

The suite covers configuration, factories, training state, validation, dataset
modes and paths, DCT shapes, synchronized augmentation, model forward, and
template-aligned submission. Tests use synthetic inputs and mocked encoders, with
no training dataset or pretrained weight download required.
