from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
import lpips

from PIL import Image, ImageFilter
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchvision.transforms.functional import to_tensor

from deep_learning.retinex_inference import enhance_image

ROOT = Path(__file__).resolve().parents[1]
LOW = ROOT / "data/finetune/Train/Low"
GT = ROOT / "data/finetune/Train/Normal"
OUT = ROOT / "results/metrics/detail_refinement.csv"

device = "cuda" if torch.cuda.is_available() else "cpu"
lpips_model = lpips.LPIPS(net="alex").to(device).eval()

files = sorted(LOW.glob("*.png"))[9::10]
strengths = [0, 25, 50, 75]
results = []

def calculate_metrics(pred, target):
    p = np.asarray(pred).astype(np.float32) / 255.0
    g = np.asarray(target).astype(np.float32) / 255.0

    psnr = peak_signal_noise_ratio(g, p, data_range=1.0)
    ssim = structural_similarity(g, p, channel_axis=2, data_range=1.0)

    with torch.inference_mode():
        a = to_tensor(pred).unsqueeze(0).to(device) * 2 - 1
        b = to_tensor(target).unsqueeze(0).to(device) * 2 - 1
        perceptual = lpips_model(a, b).item()

    return psnr, ssim, perceptual

for i, path in enumerate(files, 1):
    target_path = GT / path.name
    if not target_path.exists():
        print("Missing target:", target_path)
        continue

    source = Image.open(path).convert("RGB")
    target = Image.open(target_path).convert("RGB")

    start = time.perf_counter()
    restored = enhance_image(source)
    inference_ms = (time.perf_counter() - start) * 1000

    for strength in strengths:
        if strength == 0:
            refined = restored
        else:
            refined = restored.filter(
                ImageFilter.UnsharpMask(
                    radius=1.0,
                    percent=strength,
                    threshold=3
                )
            )

        psnr, ssim, perceptual = calculate_metrics(refined, target)

        results.append({
            "image": path.name,
            "strength": strength,
            "PSNR": psnr,
            "SSIM": ssim,
            "LPIPS": perceptual,
            "latency_ms": inference_ms
        })

    print(f"{i:02d}/{len(files)} completed: {path.name}")

df = pd.DataFrame(results)
summary = df.groupby("strength").agg(
    images=("image", "count"),
    PSNR=("PSNR", "mean"),
    SSIM=("SSIM", "mean"),
    LPIPS=("LPIPS", "mean"),
    latency_ms=("latency_ms", "mean")
).reset_index()

OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)
summary.to_csv(OUT.with_name("detail_refinement_summary.csv"), index=False)

print("\nDETAIL REFINEMENT RESULTS")
print(summary.round(4).to_string(index=False))
print("\nSaved:", OUT)
