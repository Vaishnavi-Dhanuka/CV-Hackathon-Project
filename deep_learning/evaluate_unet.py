
from pathlib import Path
import sys
import torch
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchvision.transforms.functional import to_tensor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deep_learning"))

from train import LowLightNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_DIR = ROOT / "data/raw/Kaggle/archive/LOL-v2/Real_captured/Test/Low"
TARGET_DIR = ROOT / "data/raw/Kaggle/archive/LOL-v2/Real_captured/Test/Normal"
CHECKPOINT = ROOT / "results/checkpoints/best_model.pth"

model = LowLightNet().to(DEVICE)
model.load_state_dict(
    torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True)
)
model.eval()

psnr_scores = []
ssim_scores = []

with torch.inference_mode():
    for input_path in sorted(INPUT_DIR.glob("*.png")):
        target_path = TARGET_DIR / input_path.name.replace("low", "normal", 1)

        if not target_path.exists():
            raise FileNotFoundError(target_path)

        image = Image.open(input_path).convert("RGB")
        target = np.asarray(Image.open(target_path).convert("RGB"))

        tensor = to_tensor(image).unsqueeze(0).to(DEVICE)
        h, w = tensor.shape[-2:]

        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8

        if pad_h or pad_w:
            import torch.nn.functional as F
            tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")

        output = model(tensor)[:, :, :h, :w].clamp(0, 1)
        prediction = (
            output.squeeze(0).cpu().permute(1, 2, 0).numpy() * 255
        ).round().clip(0, 255).astype(np.uint8)

        psnr_scores.append(
            peak_signal_noise_ratio(target, prediction, data_range=255)
        )
        ssim_scores.append(
            structural_similarity(
                target, prediction, channel_axis=2, data_range=255
            )
        )

print("Images:", len(psnr_scores))
print(f"U-Net PSNR: {np.mean(psnr_scores):.4f} dB")
print(f"U-Net SSIM: {np.mean(ssim_scores):.4f}")
