import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from classical.preprocessing import prepare_image, load_image
from classical.retinex import retinex_enhance

path = sys.argv[1]
img, _ = prepare_image(path)

high_path = path.replace(os.sep + "low" + os.sep, os.sep + "high" + os.sep)
gt = load_image(high_path) if (high_path != path and os.path.exists(high_path)) else None
if gt is not None and gt.shape != img.shape:
    gt = None

results = {"input": img}
configs = [("SSR", dict(method="ssr")),
           ("MSR", dict(method="msr")),
           ("guided g=0.5", dict(method="guided", gamma=0.5)),
           ("guided g=0.4", dict(method="guided", gamma=0.4))]
for name, kw in configs:
    results[name] = retinex_enhance(img, **kw)
if gt is not None:
    results["ground truth"] = gt

for name, im in results.items():
    line = f"{name:14s} mean={im.mean():6.1f}"
    if gt is not None and name != "ground truth":
        line += f"   PSNR={psnr(gt, im):5.2f}   SSIM={ssim(gt, im, channel_axis=2):.3f}"
    print(line)

fig, axes = plt.subplots(2, 3, figsize=(15, 7))
for a in axes.ravel():
    a.axis("off")
for a, (name, im) in zip(axes.ravel(), results.items()):
    a.imshow(im)
    a.set_title(name)
plt.tight_layout()
os.makedirs("results", exist_ok=True)
plt.savefig("results/retinex_demo.png", dpi=110)
plt.show()