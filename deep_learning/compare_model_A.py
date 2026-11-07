
from pathlib import Path
import sys
import csv
import torch
import numpy as np
import torch.nn.functional as F
import lpips

from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "third_party" / "Retinexformer"
CONFIG = REPO / "Options" / "RetinexFormer_LOL_v2_real.yml"

CHECKPOINTS = {
    "Baseline": ROOT / "results/checkpoints/official_only/retinexformer_official_baseline.pth",
    "Model_A": ROOT / "results/checkpoints/model_A/retinexformer_model_A_iter1000.pth",
}

LOW_DIR = ROOT / "data/model_datasets/Model_A/Val/Low"
GT_DIR = ROOT / "data/model_datasets/Model_A/Val/Normal"
OUT_DIR = ROOT / "results/metrics/Model_A/comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO))
from basicsr.models import create_model
from basicsr.utils.options import parse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_fn = lpips.LPIPS(net="alex").to(device).eval()

opt = parse(str(CONFIG), is_train=False)
opt["dist"] = False
opt["num_gpu"] = 1 if device.type == "cuda" else 0

model = create_model(opt).net_g.to(device).eval()

def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["params"], strict=True)

def infer(image):
    tensor = to_tensor(image).unsqueeze(0).to(device)
    h, w = tensor.shape[-2:]
    pad_h, pad_w = (4-h%4)%4, (4-w%4)%4

    if pad_h or pad_w:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")

    with torch.inference_mode():
        prediction = model(tensor)[:, :, :h, :w].clamp(0, 1)

    return prediction

results = []

for model_name, checkpoint_path in CHECKPOINTS.items():
    print(f"\nEvaluating {model_name}...")
    load_checkpoint(checkpoint_path)

    scores = []

    for low_path in sorted(LOW_DIR.glob("*.png")):
        gt_path = GT_DIR / low_path.name

        low = Image.open(low_path).convert("RGB")
        gt = np.asarray(Image.open(gt_path).convert("RGB"))

        prediction = infer(low)
        pred_np = np.asarray(to_pil_image(prediction.squeeze(0).cpu()))

        psnr = peak_signal_noise_ratio(gt, pred_np, data_range=255)
        ssim = structural_similarity(
            gt, pred_np, channel_axis=2, data_range=255
        )

        gt_tensor = to_tensor(Image.fromarray(gt)).unsqueeze(0).to(device) * 2 - 1
        pred_lpips = prediction * 2 - 1

        with torch.inference_mode():
            lpips_score = loss_fn(pred_lpips, gt_tensor).item()

        scores.append((low_path.name, psnr, ssim, lpips_score))

        if model_name == "Model_A":
            Image.fromarray(pred_np).save(OUT_DIR / low_path.name)

    mean_psnr = np.mean([x[1] for x in scores])
    mean_ssim = np.mean([x[2] for x in scores])
    mean_lpips = np.mean([x[3] for x in scores])

    results.append((model_name, len(scores), mean_psnr, mean_ssim, mean_lpips))

csv_path = OUT_DIR / "baseline_vs_model_A.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Model", "Images", "PSNR", "SSIM", "LPIPS"])
    writer.writerows(results)

print("\nFINAL COMPARISON")
print(f"{'Model':<12} {'PSNR':>10} {'SSIM':>10} {'LPIPS':>10}")
for name, count, psnr, ssim, lpips_score in results:
    print(f"{name:<12} {psnr:>10.4f} {ssim:>10.4f} {lpips_score:>10.4f}")

print(f"\nCSV saved: {csv_path}")
