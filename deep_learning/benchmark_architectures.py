
import csv
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import lpips
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchvision.transforms.functional import to_tensor, to_pil_image

ROOT = Path(__file__).resolve().parents[1]
VAL_LOW = ROOT / "data/model_datasets/Model_A/Val/Low"
VAL_GT = ROOT / "data/model_datasets/Model_A/Val/Normal"
OUT = ROOT / "results/metrics"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(name):
    if name == "Retinexformer":
        sys.path.insert(0, str(ROOT))
        from deep_learning.retinex_inference import _load_model, enhance_image
        model, device = _load_model()
        return model, device, lambda im: enhance_image(im)

    if name == "Multinex":
        repo = ROOT / "third_party/multinex"
        sys.path.insert(0, str(repo))
        from basicsr.models import create_model
        from basicsr.utils.options import parse

        config = repo / "Options/Multinex_LOL-v2-real.yaml"
        checkpoint = repo / "pretrained_weights/Multinex_LOL-v2-real.pth"
        opt = parse(str(config), is_train=False)
        opt["dist"] = False
        model = create_model(opt).net_g
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt.get("params", ckpt), strict=True)
        model = model.to(DEVICE).eval()

        def predict(im):
            x = to_tensor(im.convert("RGB")).unsqueeze(0).to(DEVICE)
            h, w = x.shape[-2:]
            ph, pw = (2 - h % 2) % 2, (2 - w % 2) % 2
            if ph or pw:
                x = F.pad(x, (0, pw, 0, ph), mode="reflect")
            with torch.inference_mode():
                y = model(x)
            if isinstance(y, (tuple, list)):
                y = y[0]
            return to_pil_image(
                y[:, :, :h, :w].squeeze(0).cpu().clamp(0, 1)
            )

        return model, DEVICE, predict

    if name == "Zero-DCE++":
        repo = ROOT / "third_party/Zero-DCE-PlusPlus/Zero-DCE++"
        sys.path.insert(0, str(repo))
        import model as zero_model

        net = zero_model.enhance_net_nopool(12).to(DEVICE)
        checkpoint = repo / "snapshots_Zero_DCE++/Epoch99.pth"
        net.load_state_dict(
            torch.load(checkpoint, map_location=DEVICE, weights_only=True)
        )
        net.eval()

        def predict(im):
            x = to_tensor(im.convert("RGB")).unsqueeze(0).to(DEVICE)
            h, w = x.shape[-2:]
            ph, pw = (12 - h % 12) % 12, (12 - w % 12) % 12
            if ph or pw:
                x = F.pad(x, (0, pw, 0, ph), mode="reflect")
            with torch.inference_mode():
                y, _ = net(x)
            y = y[:, :, :h, :w].squeeze(0).cpu().clamp(0, 1)
            return to_pil_image(y)

        return net, DEVICE, predict

    raise ValueError(f"Unknown model: {name}")


def run_worker(name):
    files = sorted(
        p for p in VAL_LOW.iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg")
    )
    if len(files) != 69:
        raise RuntimeError(f"Expected 69 validation images, found {len(files)}")

    model, device, predict = load_model(name)
    lpips_model = lpips.LPIPS(net="alex").to(device).eval()

    print(f"Model: {name}; device: {device}; images: {len(files)}")

    # Warm-up
    for path in files[:3]:
        with Image.open(path) as im:
            predict(im.convert("RGB"))
    if device.type == "cuda":
        torch.cuda.synchronize()

    rows = []
    for i, low_path in enumerate(files, 1):
        gt_path = VAL_GT / low_path.name
        if not gt_path.exists():
            raise FileNotFoundError(gt_path)

        with Image.open(low_path) as im:
            low = im.convert("RGB")
        with Image.open(gt_path) as im:
            gt_img = im.convert("RGB")

        if low.size != gt_img.size:
            raise ValueError(f"Pair dimensions differ: {low_path.name}")

        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        pred_img = predict(low).convert("RGB")
        if device.type == "cuda":
            torch.cuda.synchronize()
        latency = (time.perf_counter() - start) * 1000

        if pred_img.size != gt_img.size:
            raise ValueError(f"{name} output size mismatch: {low_path.name}")

        pred = np.asarray(pred_img, dtype=np.float32) / 255.0
        gt = np.asarray(gt_img, dtype=np.float32) / 255.0

        psnr = peak_signal_noise_ratio(gt, pred, data_range=1.0)
        ssim = structural_similarity(
            gt, pred, channel_axis=2, data_range=1.0
        )

        a = to_tensor(pred_img).unsqueeze(0).to(device) * 2 - 1
        b = to_tensor(gt_img).unsqueeze(0).to(device) * 2 - 1
        with torch.inference_mode():
            perceptual = lpips_model(a, b).item()

        rows.append({
            "model": name,
            "image": low_path.name,
            "PSNR": psnr,
            "SSIM": ssim,
            "LPIPS": perceptual,
            "latency_ms": latency,
        })
        print(
            f"{i}/{len(files)} PSNR={psnr:.3f} "
            f"SSIM={ssim:.4f} LPIPS={perceptual:.4f} "
            f"latency={latency:.1f}ms"
        )

    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / f"architecture_{name.lower().replace('-', '_')}.csv"
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {output}")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        run_worker(sys.argv[2])
        return

    OUT.mkdir(parents=True, exist_ok=True)
    candidates = ["Retinexformer", "Multinex", "Zero-DCE++"]

    for name in candidates:
        print(f"\n{'=' * 20} {name} {'=' * 20}", flush=True)
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker", name],
            cwd=ROOT,
            check=True,
        )

    import pandas as pd

    all_rows = []
    for name in candidates:
        filename = name.lower().replace("-", "_")
        path = OUT / f"architecture_{filename}.csv"
        all_rows.extend(pd.read_csv(path).to_dict("records"))

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT / "architecture_comparison_per_image.csv", index=False)

    summary = df.groupby("model").agg(
        images=("image", "count"),
        PSNR=("PSNR", "mean"),
        SSIM=("SSIM", "mean"),
        LPIPS=("LPIPS", "mean"),
        latency_ms=("latency_ms", "mean"),
    ).reset_index()

    summary.to_csv(OUT / "architecture_comparison_summary.csv", index=False)
    print("\nFINAL ARCHITECTURE COMPARISON")
    print(summary.round(4).to_string(index=False))
    print(f"\nSaved in: {OUT}")


if __name__ == "__main__":
    main()
