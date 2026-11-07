
from pathlib import Path
import sys
import csv
import torch
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchvision.transforms.functional import to_tensor, to_pil_image
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "third_party" / "Retinexformer"
CONFIG = REPO / "Options" / "RetinexFormer_LOL_v2_real.yml"
CHECKPOINT = ROOT / "results/checkpoints/official_only/retinexformer_official_baseline.pth"

sys.path.insert(0, str(REPO))
from basicsr.models import create_model
from basicsr.utils.options import parse


def evaluate(dataset_name):
    base = ROOT / "data/model_datasets" / dataset_name / "Val"
    input_dir = base / "Low"
    target_dir = base / "Normal"
    output_dir = ROOT / "results/metrics" / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt = parse(str(CONFIG), is_train=False)
    opt["dist"] = False
    opt["num_gpu"] = 1 if device.type == "cuda" else 0

    model = create_model(opt).net_g
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["params"], strict=True)
    model = model.to(device).eval()

    scores = []

    with torch.inference_mode():
        for index, low_path in enumerate(sorted(input_dir.glob("*.png")), 1):
            gt_path = target_dir / low_path.name
            if not gt_path.exists():
                raise FileNotFoundError(gt_path)

            image = Image.open(low_path).convert("RGB")
            target = np.asarray(Image.open(gt_path).convert("RGB"))

            tensor = to_tensor(image).unsqueeze(0).to(device)
            h, w = tensor.shape[-2:]
            pad_h, pad_w = (4-h % 4) % 4, (4-w % 4) % 4

            if pad_h or pad_w:
                tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")

            prediction = model(tensor)[:, :, :h, :w].clamp(0, 1)
            prediction = np.asarray(to_pil_image(prediction.squeeze(0).cpu()))

            psnr = peak_signal_noise_ratio(target, prediction, data_range=255)
            ssim = structural_similarity(
                target, prediction, channel_axis=2, data_range=255
            )

            scores.append((low_path.name, psnr, ssim))
            Image.fromarray(prediction).save(output_dir / low_path.name)

            print(f"[{index}] {low_path.name}: PSNR {psnr:.3f} | SSIM {ssim:.4f}")

    if not scores:
        raise RuntimeError(f"No validation images found in {input_dir}")

    csv_path = output_dir / "baseline_metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "psnr", "ssim"])
        writer.writerows(scores)

    print(f"\n{dataset_name} baseline")
    print(f"Images: {len(scores)}")
    print(f"PSNR: {np.mean([x[1] for x in scores]):.4f} dB")
    print(f"SSIM: {np.mean([x[2] for x in scores]):.4f}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    for dataset in ("Model_A", "Model_B"):
        evaluate(dataset)
