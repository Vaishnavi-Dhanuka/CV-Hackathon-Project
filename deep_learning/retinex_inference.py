
from functools import lru_cache
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "third_party" / "Retinexformer"
CONFIG = REPO / "Options" / "RetinexFormer_LOL_v2_real.yml"
CHECKPOINT = ROOT / "results" / "checkpoints" / "official_only" / "retinexformer_official_baseline.pth"

sys.path.insert(0, str(REPO))


@lru_cache(maxsize=1)
def _load_model():
    from basicsr.models import create_model
    from basicsr.utils.options import parse

    if not CONFIG.exists():
        raise FileNotFoundError(f"Retinexformer config not found: {CONFIG}")

    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Retinexformer checkpoint not found: {CHECKPOINT}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    opt = parse(str(CONFIG), is_train=False)
    opt["dist"] = False
    opt["num_gpu"] = torch.cuda.device_count() if device.type == "cuda" else 0

    model = create_model(opt).net_g

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False
    )

    model.load_state_dict(checkpoint["params"], strict=True)
    model = model.to(device).eval()

    return model, device


def enhance_image(image: Image.Image) -> Image.Image:
    """
    Enhance a PIL image using the pretrained Retinexformer model.

    Args:
        image: Input image in any PIL-supported mode.

    Returns:
        Enhanced RGB PIL image with the original dimensions.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image")

    model, device = _load_model()

    image = image.convert("RGB")
    tensor = to_tensor(image).unsqueeze(0).to(device)

    height, width = tensor.shape[-2:]
    pad_h = (4 - height % 4) % 4
    pad_w = (4 - width % 4) % 4

    if pad_h or pad_w:
        tensor = F.pad(
            tensor,
            (0, pad_w, 0, pad_h),
            mode="reflect"
        )

    with torch.inference_mode():
        output = model(tensor)

    output = output[:, :, :height, :width]
    output = output.squeeze(0).cpu().clamp(0, 1)

    return to_pil_image(output)


def enhance_file(input_path, output_path):
    """Enhance an image file and save the result."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(input_path) as image:
        enhanced = enhance_image(image)

    enhanced.save(output_path)
    return output_path
