from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
import lpips
from PIL import Image
from scipy.ndimage import gaussian_filter, median_filter
from skimage import color, exposure
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "third_party" / "Retinexformer"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ROOT / "deep_learning"))

TRAIN_LOW = ROOT / "data/finetune/Train/Low"
TRAIN_GT = ROOT / "data/finetune/Train/Normal"
OUT_DIR = ROOT / "results/evaluation_outputs/adaptive_validation"
METRICS_DIR = ROOT / "results/metrics"
CONFIG = REPO / "Options" / "RetinexFormer_LOL_v2_real.yml"
CHECKPOINT = REPO / "pretrained_weights" / "LOL_v2_real.pth"

def load_retinexformer(device):
    from basicsr.models import create_model
    from basicsr.utils.options import parse
    opt = parse(str(CONFIG), is_train=False)
    opt["dist"] = False
    opt["num_gpu"] = 1 if device.type == "cuda" else 0
    model = create_model(opt).net_g
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["params"], strict=True)
    return model.to(device).eval()

def enhance_deep(model, image, device):
    import torch.nn.functional as F
    from torchvision.transforms.functional import to_tensor, to_pil_image
    x = to_tensor(image.convert("RGB")).unsqueeze(0).to(device)
    h, w = x.shape[-2:]
    ph, pw = (4 - h % 4) % 4, (4 - w % 4) % 4
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    with torch.inference_mode():
        y = model(x)
    y = y[:, :, :h, :w].squeeze(0).clamp(0, 1).cpu()
    return np.asarray(to_pil_image(y), dtype=np.uint8)

def clahe_rgb(image):
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    lab = color.rgb2lab(rgb)
    l = np.clip(lab[..., 0] / 100.0, 0, 1)
    l2 = exposure.equalize_adapthist(l, kernel_size=64, clip_limit=0.01, nbins=256)
    lab[..., 0] = l2 * 100.0
    return np.clip(color.lab2rgb(lab) * 255.0 + 0.5, 0, 255).astype(np.uint8)

def adaptive_fusions(source_rgb, deep_rgb, classical_rgb):
    src = source_rgb.astype(np.float32) / 255.0
    deep = deep_rgb.astype(np.float32)
    classical = classical_rgb.astype(np.float32)

    # Luminance and local illumination estimate.
    y = 0.2126 * src[..., 0] + 0.7152 * src[..., 1] + 0.0722 * src[..., 2]
    illumination = gaussian_filter(y, sigma=12)
    darkness = np.clip((0.42 - illumination) / 0.42, 0, 1)

    # Robust local high-frequency residual as a noise proxy.
    local_median = median_filter(y, size=3)
    residual = np.abs(y - local_median)
    noise = gaussian_filter(residual, sigma=5)
    noise_risk = np.clip(noise / 0.055, 0, 1)

    # Edge strength protects fine structures from excessive classical blending.
    gx = np.gradient(y, axis=1)
    gy = np.gradient(y, axis=0)
    edge = np.sqrt(gx * gx + gy * gy)
    edge_risk = np.clip(gaussian_filter(edge, sigma=1) / 0.08, 0, 1)

    # Classical contribution is limited to 10%, and reduced in noisy/edge regions.
    classical_weight = 0.02 + 0.08 * darkness * (1.0 - noise_risk) * (1.0 - 0.65 * edge_risk)
    classical_weight = gaussian_filter(classical_weight, sigma=2)
    classical_weight = np.clip(classical_weight, 0.02, 0.10)

    # Ablation variants: fixed fusion, illumination-only, and illumination+noise+edge.
    illum_only = 0.02 + 0.08 * darkness
    illum_only = gaussian_filter(illum_only, sigma=2)
    illum_only = np.clip(illum_only, 0.02, 0.10)

    outputs = {
        "Retinexformer": deep_rgb,
        "Fixed_98_02": np.clip(deep * 0.98 + classical * 0.02, 0, 255).round().astype(np.uint8),
        "Adaptive_illumination": np.clip(
            deep * (1 - illum_only[..., None]) + classical * illum_only[..., None],
            0, 255
        ).round().astype(np.uint8),
        "Adaptive_illumination_noise_edge": np.clip(
            deep * (1 - classical_weight[..., None]) + classical * classical_weight[..., None],
            0, 255
        ).round().astype(np.uint8),
    }
    return outputs

def main():
    if not TRAIN_LOW.exists() or not TRAIN_GT.exists():
        raise FileNotFoundError(
            f"Expected prepared training folders: {TRAIN_LOW} and {TRAIN_GT}"
        )
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Retinexformer checkpoint missing: {CHECKPOINT}")

    all_paths = sorted(TRAIN_LOW.glob("*.png"))
    if len(all_paths) < 20:
        raise RuntimeError(f"Expected training pairs in {TRAIN_LOW}; found {len(all_paths)}")

    # Deterministic 10% validation subset. Do not train or tune on these images.
    stride = max(1, round(len(all_paths) / 70))
    val_paths = all_paths[::stride][:max(1, len(all_paths) // 10)]
    print(f"Validation images: {len(val_paths)} / {len(all_paths)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    model = load_retinexformer(device)
    lpips_model = lpips.LPIPS(net="alex").to(device).eval()

    rows = []
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    for idx, low_path in enumerate(val_paths, 1):
        gt_name = low_path.name.replace("low", "normal", 1)
        gt_path = TRAIN_GT / gt_name
        if not gt_path.exists():
            print("Skipping missing pair:", low_path.name)
            continue

        with Image.open(low_path) as im:
            source_image = im.convert("RGB")
        with Image.open(gt_path) as im:
            target = np.asarray(im.convert("RGB"), dtype=np.uint8)

        source = np.asarray(source_image, dtype=np.uint8)
        deep = enhance_deep(model, source_image, device)
        classical = clahe_rgb(source_image)
        outputs = adaptive_fusions(source, deep, classical)

        for method, pred in outputs.items():
            psnr = peak_signal_noise_ratio(target, pred, data_range=255)
            ssim = structural_similarity(target, pred, channel_axis=2, data_range=255)
            pt = torch.from_numpy(pred.copy()).permute(2, 0, 1).float()[None] / 127.5 - 1
            tt = torch.from_numpy(target.copy()).permute(2, 0, 1).float()[None] / 127.5 - 1
            with torch.inference_mode():
                lp = float(lpips_model(pt.to(device), tt.to(device)).mean().item())

            rows.append({
                "image": low_path.name,
                "method": method,
                "psnr": float(psnr),
                "ssim": float(ssim),
                "lpips_alex": lp,
            })

            if method != "Retinexformer":
                folder = OUT_DIR / method
                folder.mkdir(parents=True, exist_ok=True)
                Image.fromarray(pred).save(folder / low_path.name)

        print(f"[{idx}/{len(val_paths)}] {low_path.name}")

    detail = pd.DataFrame(rows)
    detail_path = METRICS_DIR / "adaptive_validation_per_image.csv"
    summary_path = METRICS_DIR / "adaptive_validation_summary.csv"
    detail.to_csv(detail_path, index=False)
    summary = detail.groupby("method", sort=False).agg(
        images=("image", "nunique"),
        psnr_mean=("psnr", "mean"),
        psnr_std=("psnr", "std"),
        ssim_mean=("ssim", "mean"),
        ssim_std=("ssim", "std"),
        lpips_alex_mean=("lpips_alex", "mean"),
    ).reset_index()
    summary.to_csv(summary_path, index=False)

    print("\n=== Adaptive hybrid validation results ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nSaved:", summary_path)
    print("Saved:", detail_path)
    print("Outputs:", OUT_DIR)

if __name__ == "__main__":
    main()
