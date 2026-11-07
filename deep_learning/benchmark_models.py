
import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchvision.transforms.functional import to_tensor, to_pil_image

ROOT = Path(__file__).resolve().parents[1]
LOW_DIR = ROOT / "data/finetune/Train/Low"
GT_DIR = ROOT / "data/finetune/Train/Normal"
METRICS_DIR = ROOT / "results/metrics"


def load_model(name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if name == "Retinexformer":
        sys.path.insert(0, str(ROOT))
        from deep_learning.retinex_inference import _load_model
        model, device = _load_model()

        def predict(image):
            from deep_learning.retinex_inference import enhance_image
            return enhance_image(image)

        return model, device, predict

    repo = ROOT / "third_party/multinex"
    sys.path.insert(0, str(repo))

    from basicsr.models import create_model
    from basicsr.utils.options import parse

    variant = "-Nano" if name == "Multinex-Nano" else ""
    config = repo / "Options" / f"Multinex{variant}_LOL-v2-real.yaml"
    checkpoint_path = repo / "pretrained_weights" / f"Multinex{variant}_LOL-v2-real.pth"

    if not config.exists() or not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing Multinex config/checkpoint: {config}")

    opt = parse(str(config), is_train=False)
    opt["dist"] = False
    model = create_model(opt).net_g

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("params", checkpoint)
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()

    def predict(image):
        tensor = to_tensor(image.convert("RGB")).unsqueeze(0).to(device)
        h, w = tensor.shape[-2:]
        pad_h, pad_w = (2 - h % 2) % 2, (2 - w % 2) % 2

        if pad_h or pad_w:
            tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")

        with torch.inference_mode():
            output = model(tensor)

        if isinstance(output, (tuple, list)):
            output = output[0]

        output = output[:, :, :h, :w].squeeze(0).cpu().clamp(0, 1)
        return to_pil_image(output)

    return model, device, predict


def run_worker(name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, device, predict = load_model(name)

    lpips_model = None
    try:
        import lpips
        lpips_model = lpips.LPIPS(net="alex").to(device).eval()
    except Exception as exc:
        print(f"LPIPS unavailable: {exc}")

    files = sorted(
        p for p in LOW_DIR.iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg")
    )[9::10]

    if len(files) != 68:
        raise RuntimeError(f"Expected 68 validation images, found {len(files)}")

    rows = []

    # Warm-up; not included in latency measurements.
    for path in files[:3]:
        with Image.open(path) as im:
            predict(im.convert("RGB"))
    if device.type == "cuda":
        torch.cuda.synchronize()

    for index, low_path in enumerate(files, 1):
        gt_path = GT_DIR / low_path.name
        if not gt_path.exists():
            raise FileNotFoundError(f"Missing paired ground truth: {gt_path}")

        with Image.open(low_path) as im:
            low = im.convert("RGB")
        with Image.open(gt_path) as im:
            gt = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0

        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        enhanced = predict(low)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - start) * 1000

        pred = np.asarray(enhanced.convert("RGB"), dtype=np.float32) / 255.0

        psnr = peak_signal_noise_ratio(gt, pred, data_range=1.0)
        ssim = structural_similarity(gt, pred, channel_axis=2, data_range=1.0)

        lpips_value = float("nan")
        if lpips_model is not None:
            a = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0).to(device) * 2 - 1
            b = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0).to(device) * 2 - 1
            with torch.inference_mode():
                lpips_value = lpips_model(a, b).item()

        rows.append({
            "model": name,
            "image": low_path.name,
            "PSNR": psnr,
            "SSIM": ssim,
            "LPIPS": lpips_value,
            "latency_ms": latency_ms,
        })

        print(
            f"{name}: {index:02d}/68 | "
            f"PSNR {psnr:.3f} | SSIM {ssim:.4f} | "
            f"LPIPS {lpips_value:.4f} | {latency_ms:.1f} ms"
        )

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    output = METRICS_DIR / f"benchmark_{name.lower().replace('-', '_')}.csv"

    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved: {output}")


def combine_results():
    import pandas as pd

    files = sorted(METRICS_DIR.glob("benchmark_*.csv"))
    if len(files) < 3:
        print("Run all three model benchmarks before combining.")
        return

    data = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    data.to_csv(METRICS_DIR / "model_comparison_per_image.csv", index=False)

    summary = data.groupby("model").agg(
        images=("image", "count"),
        PSNR=("PSNR", "mean"),
        SSIM=("SSIM", "mean"),
        LPIPS=("LPIPS", "mean"),
        latency_ms=("latency_ms", "mean"),
    ).reset_index()

    summary.to_csv(METRICS_DIR / "model_comparison_summary.csv", index=False)
    print("\nMODEL COMPARISON")
    print(summary.to_string(index=False))
    print(f"\nSaved results in: {METRICS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=[
        "Retinexformer", "Multinex", "Multinex-Nano"
    ])
    args = parser.parse_args()

    if args.worker:
        run_worker(args.worker)
    else:
        for name in ("Retinexformer", "Multinex", "Multinex-Nano"):
            print(f"\n{'=' * 55}\nBenchmarking {name}\n{'=' * 55}")
            subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--worker", name],
                cwd=ROOT,
                check=True,
            )
        combine_results()
