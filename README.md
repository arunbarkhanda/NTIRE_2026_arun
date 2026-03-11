Base Model: https://drive.google.com/file/d/1PbDAGqiGrwE_VmKxPDKfJVSLArwrQE80/view?usp=sharing

Refiner Model: https://drive.google.com/file/d/1Dou9LGloFVxNPc5zXSHqN6TX8VOHrEQQ/view?usp=sharing

----------------------------------------------------------------

NTIRE 2026 Image Denoising Challenge\\
Team: Variational Vision | CodaBench: arunbarkhanda\\
Team Leader: Arun Barkhanda\\
Affiliation: Clarkson University

----------------------------------------------------------------

## Overview

This repository contains the inference code and pretrained models for our submission to the NTIRE 2026 Image Denoising Challenge (σ = 50).

Our approach uses a two-stage denoising framework:
    1. **Base Attention U-Net** — initial denoising stage
    2. **Residual refinement network** — predicts a correction to restore fine textures

The final denoised image is obtained by:

```
x_final = clip(x_base + R(x_base), 0, 1)
```

where `x_base` is the base denoised image and `R(·)` is the residual prediction network.

Inference uses:
    • Patch-based processing with overlapping windows and Gaussian blending
    • 8-way test-time augmentation (4 rotations × 2 flips)

----------------------------------------------------------------

## Repository Structure
```
ntire2026-variational-vision/
├── factsheet/
│   ├── NTIRE_2026_Image_Denoising.pdf               # Compiled factsheet (PDF)
│   └── factsheet_base_residual.tex                  # LaTeX source file
│
├── inference/
│   └── ntire_test_inference.py                      # Main inference script (base-only + base+residual)
│
├── training/
│   ├── base_denoise.py                              # Training script for base Attention U-Net
│   ├── precompute_den1_for_residual_training.py     # Precomputes base model outputs for residual training
│   └── train_base_residual.py                       # Training script for residual refinement network
│
├── LICENSE
└── README.md
```

----------------------------------------------------------------

## Environment

Tested on:
    • **Lambda Labs A100** (primary): Ubuntu 22.04, TensorFlow 2.19, Keras 3, Python 3.10
    • **MacBook Air M-series** (local testing): macOS, TensorFlow 2.x

### Install dependencies

```bash
pip install pillow "numpy<2.0" --break-system-packages
```

> TensorFlow is pre-installed on Lambda GPU instances. Verify with:
> `python3 -c "import tensorflow as tf; print(tf.__version__)"`

----------------------------------------------------------------

## Download Pretrained Models

The pretrained models are hosted on Google Drive. Install `gdown` if needed:

```bash
pip install gdown
```

Download models:

```bash
gdown "https://drive.google.com/file/d/1PbDAGqiGrwE_VmKxPDKfJVSLArwrQE80/view?usp=sharing"   # ntire_unet_v7.keras
gdown "https://drive.google.com/file/d/1Dou9LGloFVxNPc5zXSHqN6TX8VOHrEQQ/view?usp=sharing"   # ntire_refiner_v7.1.keras
```

Place both `.keras` files in the root directory of the repository.

----------------------------------------------------------------

## Input Data Format

    • Images should be in **PNG format, RGB**
    • Any resolution is supported (patch-based inference handles large images)
    • Place noisy input images in the directory specified in the script's `CONFIG`:

```python
"input_dir": "/path/to/noisy/images"
```

----------------------------------------------------------------

## Running Inference

Update the paths in the `CONFIG` section at the top of `inference/ntire_final_inference.py` to match your local setup:

```python
CONFIG = {
    "base_model_path":     "ntire_unet_v7.keras",
    "residual_model_path": "ntire_refiner_v7.1.keras",
    "input_dir":           "/path/to/noisy/images",
    "output_base_only":    "denoised_base_only",   # base model results
    "output_base_res":     "denoised_base_res",    # base + residual results
    "patch_size":  96,
    "overlap":     60,
    "use_tta":     True,
    "tta_mode":    8,
    "use_xla":     True,    # set False on CPU or Apple Silicon Mac
}
```

Then run:

```bash
python3 inference/ntire_final_inference.py
```

The script will:
    1. Load both pretrained models
    2. Process each image using patch-based inference with Gaussian blending
    3. Apply 8-way test-time augmentation
    4. Apply residual refinement
    5. Save results to both output folders

----------------------------------------------------------------

## Output

Two output folders are produced:

| Folder | Contents |
|--------|----------|
| `denoised_base_only/` | Base Attention U-Net results |
| `denoised_base_res/`  | Base + residual refiner results (final submission) |

----------------------------------------------------------------

## Notes

    • `use_xla: True` is recommended on NVIDIA GPUs — first image takes ~30-60s to compile, all subsequent images are faster
    • `use_xla: False` is required on Apple Silicon Mac
    • Global normalization (`/255`) is used throughout — per-image normalization degrades PSNR
    • All 8 TTA transforms contribute positively to PSNR

----------------------------------------------------------------

## Contact

For questions regarding reproduction of results:

**Arun Barkhanda**  
Clarkson University  
barkhaa@clarkson.edu
