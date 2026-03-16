Testing Phase Results: https://drive.google.com/drive/folders/13tYDPmuVwpwnxouy3S-4oCuQeqVWYlT5?usp=sharing

Base Denoise Model (05_base_denoise_unet.keras): https://drive.google.com/file/d/1PbDAGqiGrwE_VmKxPDKfJVSLArwrQE80/view?usp=sharing

Residual Refiner Model (05_residual_refiner.keras): https://drive.google.com/file/d/1Dou9LGloFVxNPc5zXSHqN6TX8VOHrEQQ/view?usp=sharing

--------------------------------------------------------------------------------------------------------------

# NTIRE 2026 Image Denoising Challenge

**Team:** 05 Variational Vision  
**CodaBench:** arunbarkhanda  
**Team Leader:** Arun Barkhanda  
**Affiliation:** Clarkson University  

--------------------------------------------------------------------------------------------------------------

## Overview

This repository contains the inference code and pretrained models for our submission to the NTIRE 2026 Image Denoising Challenge (noise level = 50).

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
│   ├── variational_vision_factsheet.pdf     # Compiled factsheet (PDF)
│   └── variatioanl_vision_tex.zip           # LaTeX source files
│
├── model_zoo/
│   └── MODEL_DOWNLOAD.md                    # Model download instructions
│
├── models/
│   ├── 05_precompute_den1_for_residual_training.py
│   ├── 05_train_base_denoise.py
│   └── 05_train_base_residual.py
│
├── test_image/
│   └── 0000082.png                          # Sample test image
│
├── test_inference.py                  # Main inference script
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
# Due to GitHub size limits, pretrained models are hosted on Google Drive.
# Download them into the root directory of the repository:

gdown "1PbDAGqiGrwE_VmKxPDKfJVSLArwrQE80" -O model_zoo/05_base_denoise_unet.keras  # base denoise model
gdown "1Dou9LGloFVxNPc5zXSHqN6TX8VOHrEQQ" -O model_zoo/05_residual_refiner.keras   # residual refiner model
```

> See `model_zoo/MODEL_DOWNLOAD.md` for full instructions.
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

Update the paths in the `CONFIG` section at the top of `test_inference.py` to match your local setup:

```python
CONFIG = {
    "base_model_path":     "model_zoo/05_base_denoise_unet.keras",
    "residual_model_path": "model_zoo/05_residual_refiner.keras",
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
python test_inference.py
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
barkhaa@clarkson.edu | arunbarkhanda@gmail.com
