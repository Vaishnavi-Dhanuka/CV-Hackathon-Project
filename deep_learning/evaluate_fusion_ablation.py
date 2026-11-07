from pathlib import Path
import numpy as np
import pandas as pd
import torch
import lpips
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

ROOT = Path(__file__).resolve().parents[1]
LOW_DIR = ROOT / "data/raw/Kaggle/archive/LOL-v2/Real_captured/Test/Low"
GT_DIR = ROOT / "data/raw/Kaggle/archive/LOL-v2/Real_captured/Test/Normal"
DEEP_DIR = ROOT / "results/evaluation_outputs/Retinexformer"
CLAHE_DIR = ROOT / "results/evaluation_outputs/CLAHE"
OUT_ROOT = ROOT / "results/evaluation_outputs/fusion_ablation"
METRICS_DIR = ROOT / "results/metrics"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

# Blend weights: deep-learning contribution; remainder is classical CLAHE.
VARIANTS = {
    "Retinexformer": 1.00,
    "Fusion_90_10": 0.90,
    "Fusion_95_05": 0.95,
    "Fusion_98_02": 0.98,
}

def load_rgb(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    perceptual = lpips.LPIPS(net="alex").to(device).eval()

    rows = []
    names = sorted(p.name for p in LOW_DIR.glob("*.png"))
    if not names:
        raise RuntimeError(f"No test images found in {LOW_DIR}")

    for index, name in enumerate(names, 1):
        target_name = name.replace("low", "normal", 1)
        source = load_rgb(LOW_DIR / name)
        target = load_rgb(GT_DIR / target_name)
        deep = load_rgb(DEEP_DIR / name)
        classical = load_rgb(CLAHE_DIR / name)

        if not (source.shape == target.shape == deep.shape == classical.shape):
            raise ValueError(f"Image dimensions differ for {name}")

        for method, deep_weight in VARIANTS.items():
            if deep_weight == 1.0:
                pred = deep
            else:
                pred = np.clip(
                    deep.astype(np.float32) * deep_weight
                    + classical.astype(np.float32) * (1.0 - deep_weight),
                    0, 255
                ).round().astype(np.uint8)

            psnr = peak_signal_noise_ratio(target, pred, data_range=255)
            ssim = structural_similarity(
                target, pred, channel_axis=2, data_range=255
            )

            pred_t = torch.from_numpy(pred.copy()).permute(2, 0, 1).float()[None]
            target_t = torch.from_numpy(target.copy()).permute(2, 0, 1).float()[None]
            pred_t = (pred_t / 127.5 - 1.0).to(device)
            target_t = (target_t / 127.5 - 1.0).to(device)
            with torch.inference_mode():
                lpips_value = float(perceptual(pred_t, target_t).mean().item())

            rows.append({
                "image": name,
                "method": method,
                "deep_weight": deep_weight,
                "classical_weight": 1.0 - deep_weight,
                "psnr": float(psnr),
                "ssim": float(ssim),
                "lpips_alex": lpips_value,
            })

            if method != "Retinexformer":
                out_dir = OUT_ROOT / method
                out_dir.mkdir(parents=True, exist_ok=True)
                Image.fromarray(pred).save(out_dir / name)

        print(f"[{index}/{len(names)}] {name}")

    detail = pd.DataFrame(rows)
    detail_path = METRICS_DIR / "fusion_ablation_per_image.csv"
    detail.to_csv(detail_path, index=False)

    summary = (
        detail.groupby("method", sort=False)
        .agg(
            images=("image", "nunique"),
            deep_weight=("deep_weight", "first"),
            classical_weight=("classical_weight", "first"),
            psnr_mean=("psnr", "mean"),
            psnr_std=("psnr", "std"),
            ssim_mean=("ssim", "mean"),
            ssim_std=("ssim", "std"),
            lpips_alex_mean=("lpips_alex", "mean"),
        )
        .reset_index()
    )
    summary_path = METRICS_DIR / "fusion_ablation_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\n=== Fusion ablation ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nSaved:", summary_path)
    print("Saved:", detail_path)
    print("Fusion images:", OUT_ROOT)

if __name__ == "__main__":
    main()
