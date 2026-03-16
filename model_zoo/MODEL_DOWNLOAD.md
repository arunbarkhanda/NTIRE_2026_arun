# Model Zoo — Team: 05 Variational Vision

This folder contains instructions for downloading the pretrained model weights 
used in the two-stage denoising pipeline.

Due to GitHub file size limits, the pretrained models are on Google Drive.

## Download Instructions

Install gdown if not already installed:
```bash
pip install gdown
```

Download both models into this folder:
```bash
cd model_zoo/
gdown "1PbDAGqiGrwE_VmKxPDKfJVSLArwrQE80"  # 05_base_denoise_unet.keras (Base Attention U-Net)
gdown "1Dou9LGloFVxNPc5zXSHqN6TX8VOHrEQQ"  # 05_residual_refiner.keras (Residual Refiner)
```

## Model Details

| File | Description |
|------|-------------|
| `05_base_denoise_unet.keras` | Base Attention U-Net  |
| `05_residual_refiner.keras` | Residual Refiner |

## Notes
- Models are in TensorFlow/Keras format (.keras)
- Trained with TensorFlow 2.19, Keras 3, Python 3.10
- Place both files in this folder before running test_inference.py
