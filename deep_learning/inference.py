
from pathlib import Path

import torch
from PIL import Image
from torchvision.utils import save_image
from torchvision.transforms.functional import to_tensor

from train import LowLightNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "results/checkpoints/best_model.pth"
INPUT_DIR = Path("data/raw/Kaggle/archive/LOL-v2/Real_captured/Test/Low")
OUTPUT_DIR = Path("results/outputs")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

model = LowLightNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()

with torch.no_grad():
    for path in sorted(INPUT_DIR.glob("*.png")):
        image = Image.open(path).convert("RGB")
        tensor = to_tensor(image).unsqueeze(0).to(DEVICE)

        enhanced = model(tensor).squeeze(0).cpu().clamp(0, 1)

        save_image(enhanced, OUTPUT_DIR / path.name)

print(f"Enhanced images saved to {OUTPUT_DIR}")
