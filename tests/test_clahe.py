import os, sys
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from classical.preprocessing import load_image
from classical.retinex import retinex_enhance
from classical.clahe import apply_clahe

path = sys.argv[1]
lo = load_image(path)
gt = load_image(path.replace(os.sep + "low" + os.sep, os.sep + "high" + os.sep))

base = retinex_enhance(lo, target_mean=0.5)
results = {"retinex only": base}
for clip in (1.0, 2.0, 3.0):
    results[f"+CLAHE clip={clip}"] = apply_clahe(base, clip, (8, 8))
results["+CLAHE clip=2 tile=4x4"] = apply_clahe(base, 2.0, (4, 4))
results["ground truth"] = gt

for k, im in results.items():
    line = f"{k:26s} mean={im.mean():6.1f}"
    if k != "ground truth":
        line += f"  PSNR={psnr(gt, im):5.2f}  SSIM={ssim(gt, im, channel_axis=2):.3f}"
    print(line)

fig, ax = plt.subplots(2, 3, figsize=(15, 7))
for a, (k, im) in zip(ax.ravel(), results.items()):
    a.imshow(im); a.set_title(k); a.axis("off")
plt.tight_layout()
os.makedirs("results", exist_ok=True)
plt.savefig("results/clahe_demo.png", dpi=110)
plt.show()