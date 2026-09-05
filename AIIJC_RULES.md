# Task description
Task is a making solution computer vision segmentation problem.

Given an input .jpg image, the task is to identify the region modified using 
AI tools and predict a mask for that region.

## Input
RGB .jpg image with shape (H, W, 3)

## Output
Binary mask of changed region. Mask has shape (H, W, 1) and must be saved as a single-channel PNG image, where a value of 0 corresponds to the background (original image) and a value of 255 to the modified area (manipulation).

Since the prediction format consists of binary masks (0 or 255), participants independently select the optimal binarization threshold.

## Metric
As a metric is using a custom **AIC Score**:

1. Dice for Positive samples $$
Dice_{pos}
=
\frac{1}{N_{pos}}
\sum_{i \in \mathrm{positive}}
\frac{2 \cdot \left|P_i \cap G_i\right|}
{\left|P_i\right| + \left|G_i\right| + 10^{-6}}
$$
2. FPR for Negative samples (where GT hasn't changes). Image is a false positive if area of predicted mask $area_j = \frac{\left|P_j\right|}{H_j \cdot W_j} \geq 0.01 \quad \text{(more 1% frame area)}$
$$
FPR_{neg}
=
\frac{1}{N_{neg}}
\sum_{j \in \mathrm{negative}}
\mathbb{1}\left[area_j \geq 0.01\right]
$$
3. Result AIC Score — harmonic mean for two terms:$$ AIC = \frac{ 2 \cdot Dice_{pos} \cdot \left(1 - FPR_{neg}\right) }{ Dice_{pos} + \left(1 - FPR_{neg}\right) } $$

Metric range is `[0, 1]`

## Data
### Train
- `train_stage1/stage1/train.csv` – table with paths;
  - `orgl_img_path` – path to original image (might be empty);
  - `chng_img_path` – path to changed image;
  - `gt_path` - path to GT-mask of changes;
- `train_stage1/stage1/train/img/` – changed images (model's input);
- `train_stage1/stage1/train/mask/` – GT-masks of changes;
- `train_stage1/stage1/train/src/` – original images (optional).

### Test
- `test_stage1/test_stage1/test.csv` — table with paths;
  - `img_path` – path to test image.
- `test_stage1/test_stage1/test_stage1_img/` — test images;
- `test_stage1/test_stage1/submission.csv` — submission template.
  - `img_path` – test image path (like in test.csv);
  - `prediction_path` – path to predicted mask.

### Other info
Masks format (gt and predicted) is PNG-images with binary pixel values 0 / 255 with size matched with input image size

## Submission
Submission is an zip-archive with a specific structure:
- `submission.csv` - table with image and prediction paths like in template
- `predictions/`- contains all test-set binary mask predictions in PNG format referenced by submission.csv.
These .csv and folder should have exactly such names.

## Limitations
- You may not use any datasets other than provided in the competition.
- Model must be lightweight and fast with the following limitations:
  - Maximum computation complexity must be <= 100 GFLOPS per image (count with torch.utils.flop_counter.FlopCounterMode)
  - Maximum time per image must be <= 50 ms per image on the following configuration:
    - GPU: H100 80 GB;
    - RAM: 1.48 TB;
    - CPU: Xeon Platinum 8358 @2.6 Ghz, 127 kernels.
- Images from the test set may be used exclusively for generating final predictions.
Any use of the test data during the training process (including pseudo-labeling, self-training, etc.) is strictly prohibited.