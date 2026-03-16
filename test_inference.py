"""
NTIRE 2026 — Batch Inference
=====================================
Denoises images and saves TWO output folders:
  1. denoised_base_only/   — base model output
  2. denoised_base_res/    — base + residual refiner output

Usage on Lambda:
  # 1. Upload script + models + noisy images
  # 2. Install deps (system Python, no venv)
  #   pip install pillow "numpy<2.0"

  # 3. Launch in background
  nohup python3 ntire_lambda_inference.py > inference.log 2>&1 &

  # 4. Monitor
  tail -f inference.log

  # 5. Check GPU
  nvidia-smi

  # 6. When done, download both folders denoised_base_only and denoised_base_res
"""

import os
# os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import time
import numpy as np
from PIL import Image
import tensorflow as tf
import warnings

warnings.filterwarnings("ignore")

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — UPDATE THESE PATHS FOR YOUR LAMBDA INSTANCE
# ═════════════════════════════════════════════════════════════════════════════

CONFIG = {
    # ── Models ────────────────────────────────────────────────────────────
    "base_model_path":     "model_zoo/05_base_denoise_unet.keras",
    "residual_model_path": "model_zoo/05_residual_refiner.keras",

    # ── Directories ───────────────────────────────────────────────────────
    # "input_dir":           "/home/ubuntu/LSDIR_DIV2K_Test_Sigma50",
    "input_dir":           "test_image",
    "output_base_res":     "denoised_base_res",   # base + residual output
    "log_path":            "denoising_log.txt",

    # ── Patch inference ───────────────────────────────────────────────────
    "patch_size":  96,
    "overlap":     60,
    "batch_size":  64, 

    # ── TTA ───────────────────────────────────────────────────────────────
    "use_tta":   True,
    "tta_mode":  8,           # 8 = 4 rotations × 2 flips

    # ── TF optimizations ──────────────────────────────────────────────────
    "use_xla": True,          # XLA works on NVIDIA GPUs
}


# ═════════════════════════════════════════════════════════════════════════════
# BLEND WINDOW
# ═════════════════════════════════════════════════════════════════════════════

_gauss_cache = {}

def gaussian_blend_window(patch_size):
    if patch_size not in _gauss_cache:
        sigma = patch_size / 8.0
        coords = np.arange(patch_size, dtype=np.float32) - patch_size // 2
        g1d = np.exp(-coords**2 / (2 * sigma**2))
        win2d = np.outer(g1d, g1d)
        _gauss_cache[patch_size] = np.maximum(win2d[..., None], 1e-3).astype(np.float32)
    return _gauss_cache[patch_size]


# ═════════════════════════════════════════════════════════════════════════════
# PREDICT FUNCTION
# ═════════════════════════════════════════════════════════════════════════════

def make_predict_fn(model, use_xla=True, name="model"):
    @tf.function(jit_compile=use_xla)
    def _call(x):
        out = model(x, training=False)
        if isinstance(out, (list, tuple)):
            out = out[0]
        return tf.cast(out, tf.float32)

    def predict_fn(batch_np, batch_size=64):
        results = []
        for i in range(0, len(batch_np), batch_size):
            chunk = tf.constant(batch_np[i:i + batch_size], dtype=tf.float32)
            results.append(_call(chunk).numpy())
        return np.concatenate(results, axis=0)

    # Warm-up (triggers XLA compilation)
    ds = model.input_shape
    if isinstance(ds, list):
        ds = ds[0]
    ps = ds[1] if ds[1] else 96
    _ = _call(tf.zeros((1, ps, ps, 3), dtype=tf.float32))
    mode = "XLA" if use_xla else "graph"
    print(f"  {name}: ready ({mode})")

    return predict_fn


# ═════════════════════════════════════════════════════════════════════════════
# PATCH-BASED INFERENCE
# ═════════════════════════════════════════════════════════════════════════════

def denoise_with_patches(predict_fn, noisy_image, patch_size, overlap, batch_size):
    h, w, c = noisy_image.shape
    stride = patch_size - overlap
    blend_weight = gaussian_blend_window(patch_size)

    ys = list(range(0, max(h - patch_size + 1, 1), stride))
    xs = list(range(0, max(w - patch_size + 1, 1), stride))
    if not ys: ys = [0]
    if not xs: xs = [0]
    if ys[-1] != max(h - patch_size, 0):
        ys.append(max(h - patch_size, 0))
    if xs[-1] != max(w - patch_size, 0):
        xs.append(max(w - patch_size, 0))

    patches, coords_list = [], []
    for y in ys:
        for x in xs:
            patch = noisy_image[y:y + patch_size, x:x + patch_size, :]
            ph, pw = patch.shape[:2]
            if ph != patch_size or pw != patch_size:
                patch = np.pad(patch,
                               ((0, patch_size - ph), (0, patch_size - pw), (0, 0)),
                               mode="reflect")
            patches.append(patch)
            coords_list.append((y, x))

    den_patches = predict_fn(np.stack(patches, axis=0).astype(np.float32),
                             batch_size=batch_size)

    denoised = np.zeros_like(noisy_image, dtype=np.float32)
    weights = np.zeros((h, w, 1), dtype=np.float32)
    for i, (y, x) in enumerate(coords_list):
        dp = np.asarray(den_patches[i], dtype=np.float32)
        rh, rw = min(patch_size, h - y), min(patch_size, w - x)
        dp = dp[:rh, :rw, :]
        bw = blend_weight[:rh, :rw, :]
        denoised[y:y + rh, x:x + rw, :] += dp * bw
        weights[y:y + rh, x:x + rw, :] += bw

    return np.clip(denoised / (weights + 1e-8), 0, 1).astype(np.float32)


# ═════════════════════════════════════════════════════════════════════════════
# TTA
# ═════════════════════════════════════════════════════════════════════════════

def _rot(img, k):
    return np.rot90(img, k=k, axes=(0, 1)).copy()


def denoise_tta(predict_fn, noisy, patch_size, overlap, batch_size, tta_mode=8):
    h, w, c = noisy.shape
    accum = np.zeros((h, w, c), dtype=np.float64)
    count = 0

    for k in (0, 1, 2, 3):
        noisy_rot = _rot(noisy, k)
        den = denoise_with_patches(predict_fn, noisy_rot, patch_size, overlap, batch_size)
        accum += _rot(den, (4 - k) % 4).astype(np.float64)
        count += 1

        if tta_mode >= 8:
            noisy_flip = np.fliplr(noisy_rot)
            den_flip = denoise_with_patches(predict_fn, noisy_flip,
                                            patch_size, overlap, batch_size)
            accum += _rot(np.fliplr(den_flip), (4 - k) % 4).astype(np.float64)
            count += 1

    return np.clip(accum / count, 0, 1).astype(np.float32)


# ═════════════════════════════════════════════════════════════════════════════
# I/O
# ═════════════════════════════════════════════════════════════════════════════

def load_image(path):
    return np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def save_image(path, img01):
    Image.fromarray((np.clip(img01, 0, 1) * 255).astype(np.uint8)).save(path)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN BATCH PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def run(cfg):
    print("=" * 70)
    print("NTIRE 2026 — Lambda Batch Inference")
    print("=" * 70)

    # ── GPU check ─────────────────────────────────────────────────────────
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"GPU: {gpus[0].name}")
    else:
        print("WARNING: No GPU detected — inference will be slow!")

    # ── Create output dirs ────────────────────────────────────────────────
    os.makedirs(cfg["output_base_res"], exist_ok=True)

    # ── Load models ───────────────────────────────────────────────────────
    print(f"\nLoading base model: {cfg['base_model_path']}")
    base_model = tf.keras.models.load_model(cfg["base_model_path"], compile=False)
    base_fn = make_predict_fn(base_model, use_xla=cfg["use_xla"], name="base")

    print(f"Loading residual model: {cfg['residual_model_path']}")
    res_model = tf.keras.models.load_model(cfg["residual_model_path"], compile=False)
    res_fn = make_predict_fn(res_model, use_xla=cfg["use_xla"], name="residual")

    # ── Collect input files ───────────────────────────────────────────────
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    all_files = sorted([
        f for f in os.listdir(cfg["input_dir"])
        if os.path.splitext(f)[1].lower() in exts
    ])

    if not all_files:
        print(f"ERROR: No images found in {cfg['input_dir']}")
        return

    print(f"\nFound {len(all_files)} images in {cfg['input_dir']}")
    tta_str = f"{cfg['tta_mode']}-way TTA" if cfg["use_tta"] else "no TTA"
    print(f"Settings: {tta_str}, patch={cfg['patch_size']}, "
        f"overlap={cfg['overlap']}, batch={cfg['batch_size']}")
    print(f"Output folder: {cfg['output_base_res']}")

    # ── Log header ────────────────────────────────────────────────────────
    log_lines = [
        "=" * 80,
        "NTIRE 2026 Image Denoising — Inference Log",
        f"GPU          : {gpus[0].name if gpus else 'CPU'}",
        f"Base model   : {cfg['base_model_path']}",
        f"Residual     : {cfg['residual_model_path']}",
        f"TTA          : {tta_str}",
        f"Patch/Overlap: {cfg['patch_size']}/{cfg['overlap']}",
        f"Batch size   : {cfg['batch_size']}",
        "=" * 80,
        f"{'#':>4}  {'Filename':<30}  {'Base(s)':>8}  {'Res(s)':>7}  "
        f"{'Total(s)':>9}  {'PSNR base':>10}  {'PSNR b+r':>10}",
        "-" * 80,
    ]

    total_start = time.time()
    ps = cfg["patch_size"]
    ov = cfg["overlap"]
    bs = cfg["batch_size"]

    # ── Process each image ────────────────────────────────────────────────
    for idx, fname in enumerate(all_files, 1):
        fpath = os.path.join(cfg["input_dir"], fname)
        print(f"  [{idx:>3}/{len(all_files)}] {fname} ... ", end="", flush=True)

        noisy = load_image(fpath)

        # ── Stage 1: Base model + TTA ─────────────────────────────────────
        t0 = time.time()
        if cfg["use_tta"]:
            den1 = denoise_tta(base_fn, noisy, ps, ov, bs, cfg["tta_mode"])
        else:
            den1 = denoise_with_patches(base_fn, noisy, ps, ov, bs)
        den1 = np.clip(den1, 0, 1).astype(np.float32)
        t_base = time.time() - t0

        # ── Stage 2: Residual refiner ─────────────────────────────────────
        t0 = time.time()
        residual = denoise_with_patches(res_fn, den1, ps, ov, bs)
        den2 = np.clip(den1 + residual, 0, 1).astype(np.float32)
        t_res = time.time() - t0

        t_total = t_base + t_res

        # ── Save to BOTH folders ──────────────────────────────────────────
        save_image(os.path.join(cfg["output_base_res"], fname), den2)

        # ── Log ───────────────────────────────────────────────────────────
        log_line = (f"{idx:>4}  {fname:<30}  {t_base:>8.2f}  "
                    f"{t_res:>7.2f}  {t_total:>9.2f}")
        log_lines.append(log_line)
        print(f"base={t_base:.1f}s  res={t_res:.1f}s  total={t_total:.1f}s")

    # ── Summary ───────────────────────────────────────────────────────────
    wall = time.time() - total_start
    log_lines.append("-" * 80)
    log_lines.append(f"Total wall-clock time : {wall:.2f}s ({wall / 60:.2f} min)")
    log_lines.append(f"Images processed      : {len(all_files)}")
    log_lines.append(f"Avg time per image    : {wall / max(len(all_files), 1):.2f}s")
    log_lines.append("=" * 80)

    with open(cfg["log_path"], "w") as f:
        f.write("\n".join(log_lines) + "\n")

    print(f"\n{'='*70}")
    print(f"DONE!")
    print(f"  Base+Res images  : {cfg['output_base_res']}/  ({len(all_files)} files)")
    print(f"  Log              : {cfg['log_path']}")
    print(f"  Wall time        : {wall:.1f}s ({wall / 60:.1f} min)")
    print(f"  Avg per image    : {wall / len(all_files):.1f}s")
    print(f"{'='*70}")


if __name__ == "__main__":
    run(CONFIG)
