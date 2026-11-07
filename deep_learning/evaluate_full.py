"""
Full low-light evaluation for LOLv2 Real Captured.
Run from project root:
  python deep_learning/evaluate_full.py

Evaluates:
  - Input (no enhancement)
  - CLAHE (classical baseline)
  - Retinexformer (official pretrained checkpoint)
  - Hybrid (adaptive blend of Retinexformer + CLAHE)

Writes per-image and summary CSVs under results/metrics and enhanced
images under results/evaluation_outputs.

LPIPS is full-reference (lower is better). NIQE is optional and is
not disk I/O or metric calculation.
"""
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from skimage import color, exposure
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchvision.transforms.functional import to_tensor, to_pil_image

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "third_party" / "Retinexformer"
CONFIG = REPO / "Options" / "RetinexFormer_LOL_v2_real.yml"
CHECKPOINT = REPO / "pretrained_weights" / "LOL_v2_real.pth"
INPUT_DIR = ROOT / "data/raw/Kaggle/archive/LOL-v2/Real_captured/Test/Low"
TARGET_DIR = ROOT / "data/raw/Kaggle/archive/LOL-v2/Real_captured/Test/Normal"
METRICS_DIR = ROOT / "results/metrics"
OUTPUT_ROOT = ROOT / "results/evaluation_outputs"
METRICS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO))
from basicsr.models import create_model
from basicsr.utils.options import parse

# LPIPS is optional but expected in this environment.
try:
    import lpips
    LPIPS_AVAILABLE = True
except Exception:
    LPIPS_AVAILABLE = False


def load_retinexformer(device):
    if not CONFIG.exists():
        raise FileNotFoundError(f"Missing config: {CONFIG}")
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Missing checkpoint: {CHECKPOINT}")

    opt = parse(str(CONFIG), is_train=False)
    opt["dist"] = False
    opt["num_gpu"] = 1 if device.type == "cuda" else 0
    model = create_model(opt).net_g
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["params"], strict=True)
    return model.to(device).eval()


def clahe_rgb(image):
    """CLAHE on LAB luminance; returns uint8 RGB."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    lab = color.rgb2lab(rgb)
    luminance = np.clip(lab[..., 0] / 100.0, 0.0, 1.0)
    enhanced_l = exposure.equalize_adapthist(
        luminance, kernel_size=64, clip_limit=0.01, nbins=256
    )
    lab[..., 0] = np.clip(enhanced_l, 0.0, 1.0) * 100.0
    out = np.clip(color.lab2rgb(lab), 0.0, 1.0)
    return (out * 255.0 + 0.5).astype(np.uint8)


def tensor_for_model(image, device):
    tensor = to_tensor(image.convert("RGB")).unsqueeze(0).to(device)
    h, w = tensor.shape[-2:]
    pad_h, pad_w = (4 - h % 4) % 4, (4 - w % 4) % 4
    if pad_h or pad_w:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
    return tensor, h, w


def run_retinex(model, image, device):
    tensor, h, w = tensor_for_model(image, device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        output = model(tensor)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    output = output[:, :, :h, :w].squeeze(0).clamp(0, 1).cpu()
    return np.asarray(to_pil_image(output)), elapsed_ms


def hybrid_adaptive(deep_rgb, clahe_rgb_image, source_rgb):
    """
    Spatially adaptive blend.
    Darker source regions receive more CLAHE; brighter regions rely more
    on Retinexformer. Weight is smoothed to avoid harsh transitions.
    """
    source = source_rgb.astype(np.float32) / 255.0
    y = 0.2126 * source[..., 0] + 0.7152 * source[..., 1] + 0.0722 * source[..., 2]
    # Deep weight rises from 0.55 in very dark regions to 0.85 in brighter regions.
    deep_weight = 0.55 + 0.30 * np.clip(y / 0.45, 0.0, 1.0)
    deep = deep_rgb.astype(np.float32)
    classical = clahe_rgb_image.astype(np.float32)
    out = deep * deep_weight[..., None] + classical * (1.0 - deep_weight[..., None])
    return np.clip(out, 0, 255).astype(np.uint8)


def score_pair(reference, prediction, lpips_model, device):
    psnr = peak_signal_noise_ratio(reference, prediction, data_range=255)
    ssim = structural_similarity(
        reference, prediction, channel_axis=2, data_range=255
    )
    lpips_value = np.nan
    if lpips_model is not None:
        pred_t = torch.from_numpy(prediction.copy()).permute(2, 0, 1).float()[None] / 127.5 - 1
        ref_t = torch.from_numpy(reference.copy()).permute(2, 0, 1).float()[None] / 127.5 - 1
        with torch.inference_mode():
            lpips_value = float(lpips_model(pred_t.to(device), ref_t.to(device)).mean().item())
    return float(psnr), float(ssim), lpips_value


def main():
    if not INPUT_DIR.exists() or not TARGET_DIR.exists():
        raise FileNotFoundError("Dataset folders not found. Check INPUT_DIR and TARGET_DIR.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"LPIPS available: {LPIPS_AVAILABLE}")
    model = load_retinexformer(device)

    # Warm-up is excluded from latency.
    warmup_image = Image.open(sorted(INPUT_DIR.glob("*.png"))[0]).convert("RGB")
    for _ in range(3):
        run_retinex(model, warmup_image, device)

    lpips_model = None
    if LPIPS_AVAILABLE:
        lpips_model = lpips.LPIPS(net="alex").to(device).eval()


    image_paths = sorted(INPUT_DIR.glob("*.png"))
    if not image_paths:
        raise RuntimeError(f"No test PNG images found in {INPUT_DIR}")

    rows = []
    for idx, input_path in enumerate(image_paths, 1):
        target_path = TARGET_DIR / input_path.name.replace("low", "normal", 1)
        if not target_path.exists():
            warnings.warn(f"Missing paired target: {target_path}")
            continue

        source_image = Image.open(input_path).convert("RGB")
        reference = np.asarray(Image.open(target_path).convert("RGB"), dtype=np.uint8)
        source = np.asarray(source_image, dtype=np.uint8)

        deep, latency_ms = run_retinex(model, source_image, device)
        classical = clahe_rgb(source_image)
        hybrid = hybrid_adaptive(deep, classical, source)

        variants = {
            "Input": source,
            "CLAHE": classical,
            "Retinexformer": deep,
            "Hybrid_Adaptive": hybrid,
        }

        for method, prediction in variants.items():
            if prediction.shape != reference.shape:
                raise ValueError(
                    f"Shape mismatch for {input_path.name}: "
                    f"{prediction.shape} vs {reference.shape}"
                )
            psnr, ssim, lpips_value = score_pair(
                reference, prediction, lpips_model, device
            )
            row = {
                "image": input_path.name,
                "method": method,
                "psnr": psnr,
                "ssim": ssim,
                "lpips_alex": lpips_value,
                "latency_ms": latency_ms if method == "Retinexformer" else np.nan,
            }
            rows.append(row)

            out_dir = OUTPUT_ROOT / method
            out_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(prediction).save(out_dir / input_path.name)

        print(f"[{idx}/{len(image_paths)}] {input_path.name} | Retinex latency {latency_ms:.2f} ms")

    per_image = pd.DataFrame(rows)
    per_image.to_csv(METRICS_DIR / "evaluation_per_image.csv", index=False)

    summary = (
        per_image.groupby("method", sort=False)
        .agg(
            images=("image", "nunique"),
            psnr_mean=("psnr", "mean"),
            psnr_std=("psnr", "std"),
            ssim_mean=("ssim", "mean"),
            ssim_std=("ssim", "std"),
            lpips_alex_mean=("lpips_alex", "mean"),
            latency_ms=("latency_ms", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(METRICS_DIR / "evaluation_summary.csv", index=False)

    # Worst 10 Retinexformer images by PSNR for failure-case inspection.
    failures = (
        per_image[per_image["method"] == "Retinexformer"]
        .nsmallest(10, "psnr")
        .sort_values("psnr")
    )
    failures.to_csv(METRICS_DIR / "failure_cases_retinexformer.csv", index=False)

    print("\n=== Evaluation summary ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved: {METRICS_DIR / 'evaluation_summary.csv'}")
    print(f"Saved: {METRICS_DIR / 'evaluation_per_image.csv'}")
    print(f"Saved: {METRICS_DIR / 'failure_cases_retinexformer.csv'}")
    print(f"Enhanced images: {OUTPUT_ROOT}")

    if not LPIPS_AVAILABLE:
        print("\nLPIPS skipped. Install with: pip install lpips")


if __name__ == "__main__":
    main()
