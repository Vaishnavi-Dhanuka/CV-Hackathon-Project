
from pathlib import Path
import sys
import yaml
import torch
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchvision.transforms.functional import to_tensor, to_pil_image
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "third_party" / "Retinexformer"
CONFIG = REPO / "Options" / "RetinexFormer_LOL_v2_real.yml"
CHECKPOINT = REPO / "pretrained_weights" / "LOL_v2_real.pth"

INPUT_DIR = ROOT / "data/raw/Kaggle/archive/LOL-v2/Real_captured/Test/Low"
TARGET_DIR = ROOT / "data/raw/Kaggle/archive/LOL-v2/Real_captured/Test/Normal"
OUTPUT_DIR = ROOT / "results/retinexformer"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO))
from basicsr.models import create_model
from basicsr.utils.options import parse


def main():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    opt = parse(str(CONFIG), is_train=False)
    opt["dist"] = False
    opt["num_gpu"] = 1 if device.type == "cuda" else 0

    model = create_model(opt).net_g

    checkpoint = torch.load(
        CHECKPOINT, map_location="cpu", weights_only=False
    )
    state = checkpoint["params"]
    model.load_state_dict(state, strict=True)

    model = model.to(device).eval()

    psnr_scores = []
    ssim_scores = []

    input_paths = sorted(INPUT_DIR.glob("*.png"))

    if not input_paths:
        raise RuntimeError(f"No PNG images found in {INPUT_DIR}")

    with torch.inference_mode():
        for index, input_path in enumerate(input_paths, 1):
            target_name = input_path.name.replace("low", "normal", 1)
            target_path = TARGET_DIR / target_name

            if not target_path.exists():
                print("Missing target:", target_path.name)
                continue

            image = Image.open(input_path).convert("RGB")
            target = np.asarray(
                Image.open(target_path).convert("RGB")
            )

            tensor = to_tensor(image).unsqueeze(0).to(device)
            h, w = tensor.shape[-2:]

            pad_h = (4 - h % 4) % 4
            pad_w = (4 - w % 4) % 4

            if pad_h or pad_w:
                tensor = F.pad(
                    tensor, (0, pad_w, 0, pad_h), mode="reflect"
                )

            output = model(tensor)
            output = output[:, :, :h, :w].clamp(0, 1)
            output = output.squeeze(0).cpu()

            prediction = np.asarray(to_pil_image(output))

            psnr = peak_signal_noise_ratio(
                target, prediction, data_range=255
            )
            ssim = structural_similarity(
                target, prediction, channel_axis=2, data_range=255
            )

            psnr_scores.append(psnr)
            ssim_scores.append(ssim)

            Image.fromarray(prediction).save(
                OUTPUT_DIR / input_path.name
            )

            print(
                f"[{index}/{len(input_paths)}] "
                f"{input_path.name} | PSNR: {psnr:.2f} | SSIM: {ssim:.4f}"
            )

    print("\nRetinexformer Results")
    print("Images:", len(psnr_scores))
    print(f"Average PSNR: {np.mean(psnr_scores):.4f} dB")
    print(f"Average SSIM: {np.mean(ssim_scores):.4f}")
    print("Outputs:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
