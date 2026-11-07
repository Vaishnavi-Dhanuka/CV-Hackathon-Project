from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips
from torchvision.transforms.functional import to_tensor

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "third_party" / "Zero-DCE-PlusPlus" / "Zero-DCE++"
LOW = ROOT / "data" / "finetune" / "Train" / "Low"
GT = ROOT / "data" / "finetune" / "Train" / "Normal"
CHECKPOINT = REPO / "snapshots_Zero_DCE++" / "Epoch99.pth"
OUTPUT = ROOT / "results" / "metrics" / "benchmark_zero_dce.csv"

sys.path.insert(0, str(REPO))
import model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = model.enhance_net_nopool(12).to(device)
state = torch.load(CHECKPOINT, map_location=device, weights_only=True)
net.load_state_dict(state)
net.eval()

lpips_model = lpips.LPIPS(net="alex").to(device).eval()

files = sorted(LOW.glob("*.png"))[9::10]
results = []

with torch.inference_mode():
    for i, path in enumerate(files, 1):
        target_path = GT / path.name
        if not target_path.exists():
            continue

        source = Image.open(path).convert("RGB")
        target = Image.open(target_path).convert("RGB")
        arr = np.asarray(source).astype(np.float32) / 255.0

        h, w = arr.shape[:2]
        h2 = (h // 12) * 12
        w2 = (w // 12) * 12
        arr = arr[:h2, :w2, :]

        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        enhanced, _ = net(tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latency = (time.perf_counter() - start) * 1000

        pred = enhanced.squeeze(0).clamp(0, 1).cpu()
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(0),
            size=(source.height, source.width),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        pred_img = Image.fromarray(
            (pred.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        )
        target = target.resize(source.size)

        p = np.asarray(pred_img).astype(np.float32) / 255.0
        g = np.asarray(target).astype(np.float32) / 255.0

        psnr = peak_signal_noise_ratio(g, p, data_range=1.0)
        ssim = structural_similarity(g, p, channel_axis=2, data_range=1.0)

        a = to_tensor(pred_img).unsqueeze(0).to(device) * 2 - 1
        b = to_tensor(target).unsqueeze(0).to(device) * 2 - 1
        perceptual = lpips_model(a, b).item()

        results.append({
            "image": path.name,
            "PSNR": psnr,
            "SSIM": ssim,
            "LPIPS": perceptual,
            "latency_ms": latency
        })

        print(f"{i:02d}/{len(files)} | PSNR {psnr:.3f} | SSIM {ssim:.4f} | LPIPS {perceptual:.4f} | {latency:.1f} ms")

df = pd.DataFrame(results)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT, index=False)

print("\nZERO-DCE++ RESULTS")
print(df[["PSNR", "SSIM", "LPIPS", "latency_ms"]].mean().round(4))
print("\nSaved:", OUTPUT)
