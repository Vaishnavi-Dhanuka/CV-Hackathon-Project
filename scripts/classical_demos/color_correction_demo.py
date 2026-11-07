import os, sys
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from classical.preprocessing import load_image
from classical.classical_pipeline import enhance_classical
from classical.color_correction import color_correct, choose_color_params, color_cast

# usage: python scripts/classical_demos/color_correction_demo.py <path to our485> [step]
root = sys.argv[1]
step = int(sys.argv[2]) if len(sys.argv) > 2 else 10
low_dir, high_dir = os.path.join(root, "low"), os.path.join(root, "high")
names = sorted(f for f in os.listdir(low_dir)
               if f.lower().endswith(".png") and not f.startswith("._"))[::step]

# ---------- Test 1: does it remove a colour cast we add ourselves? ----------
print("TEST 1: synthetic colour cast on a well-lit image")
gt = load_image(os.path.join(high_dir, names[0]))
print(f"{'cast gains (R,G,B)':20s} {'cast before':>11} {'cast after':>10} {'PSNR before':>11} {'PSNR after':>10}")
demo = None
for gains in [(1.3, 1.0, 0.75), (0.8, 1.0, 1.25), (1.15, 1.0, 0.9)]:
    bad = np.clip(gt.astype(np.float32) * gains, 0, 255).astype(np.uint8)
    params, _ = choose_color_params(bad)
    fixed = color_correct(bad, **params)
    print(f"{str(gains):20s} {color_cast(bad):11.3f} {color_cast(fixed):10.3f} "
          f"{psnr(gt, bad):11.2f} {psnr(gt, fixed):10.2f}")
    demo = demo or (bad, fixed)

# ---------- Test 2: highlight protection ----------
hl_before = float((gt.mean(axis=2) > 240).mean())
params, _ = choose_color_params(gt)
hl_after = float((color_correct(gt, **params).mean(axis=2) > 240).mean())
print(f"\nTEST 2: fraction of near-white pixels on a bright image: before {hl_before:.4f}, after {hl_after:.4f}")

# ---------- Test 3: ablation inside the full pipeline ----------
print(f"\nTEST 3: full pipeline on {len(names)} low images, colour correction off vs on")
rows = []
for n in names:
    lo = load_image(os.path.join(low_dir, n))
    ref = load_image(os.path.join(high_dir, n))
    off, _ = enhance_classical(lo, {"use_color_correction": False})
    on, info = enhance_classical(lo, {"use_color_correction": True})
    rows.append((psnr(ref, off), ssim(ref, off, channel_axis=2),
                 psnr(ref, on), ssim(ref, on, channel_axis=2)))
r = np.array(rows)
print(f"{'':12s} {'PSNR':>6} {'SSIM':>6}")
print(f"{'cc off':12s} {r[:, 0].mean():6.2f} {r[:, 1].mean():6.3f}")
print(f"{'cc on':12s} {r[:, 2].mean():6.2f} {r[:, 3].mean():6.3f}")
d = r[:, 2] - r[:, 0]
print(f"cc helps PSNR on {int((d > 0).sum())}/{len(d)} images, mean change {d.mean():+.2f} dB, worst {d.min():+.2f} dB")
print("params used on last image:", info["meta"]["config"]["params"].get("color_correction"))

os.makedirs("results", exist_ok=True)
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for a, (t, im) in zip(ax, [("with cast", demo[0]), ("corrected", demo[1]), ("original", gt)]):
    a.imshow(im); a.set_title(t); a.axis("off")
plt.tight_layout(); plt.savefig("results/color_demo.png", dpi=110); plt.show()
