Testing Phase Results: https://drive.google.com/drive/folders/1wKqC6KT3zaIKX9P-FDXR3ALFE5O2k9ST?usp=sharing

Base Model: https://drive.google.com/file/d/1PbDAGqiGrwE_VmKxPDKfJVSLArwrQE80/view?usp=sharing

Refiner Model: https://drive.google.com/file/d/1Dou9LGloFVxNPc5zXSHqN6TX8VOHrEQQ/view?usp=sharing

--------------------------------------------------------------------------------------------------------------

# NTIRE 2026 Image Denoising Challenge

**Team:** Variational Vision  
**CodaBench:** arunbarkhanda  
**Team Leader:** Arun Barkhanda  
**Affiliation:** Clarkson University  

--------------------------------------------------------------------------------------------------------------

## Overview

This repository contains the inference code and pretrained models for our submission to the NTIRE 2026 Image Denoising Challenge (\u03c3 = 50).

Our method uses a two-stage denoising framework:
1. **Base Attention U-Net** for initial image denoising
2. **Residual refinement network** to predict an additive correction for restoring fine textures

The final restored image is computed as

```
x_final = clip(x_base + R(x_base), 0, 1)
```


where `x_base` is the base denoised image and `R(·)` is the residual prediction network.

Inference uses:
    • Patch-based processing with overlapping windows and Gaussian blending
    • 8-way test-time augmentation (4 rotations × 2 flips)

--------------------------------------------------------------------------------------------------------------

## Repository Structure

```
NTIRE_2026_arun/
├── factsheet/
│   ├── NTIRE_2026_Image_Denoising.pdf               # Compiled factsheet (PDF)
│   └── factsheet_base_residual.tex                  # LaTeX source file
│
├── training/
│   ├── base_denoise.py                              # Training script for base Attention U-Net
│   ├── precompute_den1_for_residual_training.py     # Precomputes base model outputs for residual training
│   └── train_base_residual.py                       # Training script for residual refinement network
│
├── ntire_test_inference.py                          # Main inference script
├── LICENSE
└── README.md
```
--------------------------------------------------------------------------------------------------------------

## Environment Used

**Training environment:**
    • Lambda Labs A100-SXM4-40GB: Ubuntu 22.04, TensorFlow 2.19, Keras 3, Python 3.10

**Inference tested on:**
    • Lambda Labs A100 (primary): Ubuntu 22.04, TensorFlow 2.19, Keras 3, Python 3.10
    • MacBook Air M-series (local testing): macOS, TensorFlow 2.21, Keras 3, Python 3.10
    
--------------------------------------------------------------------------------------------------------------

## Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/arunbarkhanda/NTIRE_2026_arun
cd NTIRE_2026_arun

# 2. Install dependencies
pip install tensorflow pillow numpy gdown

# 3. Download pretrained models
gdown "1PbDAGqiGrwE_VmKxPDKfJVSLArwrQE80"   # ntire_unet_v7.keras (base model)
gdown "1Dou9LGloFVxNPc5zXSHqN6TX8VOHrEQQ"   # ntire_refiner_v7.1.keras (refiner model)
```

Place both `.keras` files in the root directory of the repository.

--------------------------------------------------------------------------------------------------------------

## Input Data Format

    • Images should be in **PNG format, RGB**
    • Any resolution is supported (patch-based inference handles large images)
    • Place noisy input images in the directory specified in the CONFIG section of the script:

```python
"input_dir": "/path/to/noisy/images/folder"
```

--------------------------------------------------------------------------------------------------------------

## Running Inference

Update the paths in the `CONFIG` section at the top of `ntire_test_inference.py` to match your local setup:

```python
CONFIG = {
    "base_model_path":     "ntire_unet_v7.keras",
    "residual_model_path": "ntire_refiner_v7.1.keras",
    "input_dir":           "/path/to/noisy/images/folder",
    "output_base_res":     "denoised_base_res",
    "patch_size":  96,
    "overlap":     60,
    "batch_size":  64,
    "use_tta":     True,
    "tta_mode":    8,
    "use_xla":     True,    # set False on CPU or Apple Silicon Mac
}
```

Then run:

```bash
python ntire_test_inference.py
```

The script will:
	1.	Load the pretrained base and residual models
	2.	Process each image using patch-based inference with Gaussian blending
	3.	Apply 8-way test-time augmentation
	4.	Perform residual refinement
	5.	Save the final denoised images to the output folder (denoised_base_res)

--------------------------------------------------------------------------------------------------------------

## Output

One output folder is produced:

| Folder | Contents |
|--------|----------|
| `denoised_base_res/`  | Base + residual refiner results (final submission) |

--------------------------------------------------------------------------------------------------------------

## Notes
    • `use_xla: False` is recommended on CPU
    • Global normalization (`/255`) is used throughout — per-image normalization degrades PSNR
    • All 8 TTA transforms contribute positively to PSNR
    
--------------------------------------------------------------------------------------------------------------

## Contact

For questions regarding reproduction of results:

**Arun Barkhanda**  
Clarkson University  
barkhaa@clarkson.edu
