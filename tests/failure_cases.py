import os, sys, csv
import numpy as np
import cv2
import matplotlib.pyplot as plt
from classical.preprocessing import load_image
from classical.classical_pipeline import enhance_classical

# usage: python -m tests.failure_cases <low_dir> <high_dir> <per_image_csv> [n]
low_dir, high_dir, csv_path = sys.argv[1:4]
n = int(sys.argv[4]) if len(sys.argv) > 4 else 3

rows = list(csv.DictReader(open(csv_path)))
rows.sort(key=lambda r: float(r["psnr"]))
worst = rows[:n]


def lum(x):
    return float(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).mean())


os.makedirs("results", exist_ok=True)
fig, ax = plt.subplots(n, 4, figsize=(16, 3.6 * n))
ax = np.atleast_2d(ax)
for i, r in enumerate(worst):
    lo = load_image(os.path.join(low_dir, r["low"]))
    gt = load_image(os.path.join(high_dir, r["high"]))
    out, info = enhance_classical(lo)
    m = info["meta"]["config"]
    p = m["params"]
    print(f"\n{r['low']}: PSNR {float(r['psnr']):.2f}")
    print(f"  mean brightness  input {lum(lo):6.1f}   retinex {lum(info['retinex']):6.1f}   "
          f"final {lum(out):6.1f}   ground truth {lum(gt):6.1f}")
    print(f"  adaptive rule: {m['adaptive_rule']}")
    print(f"  denoise: {p['denoise']}")
    print(f"  clahe applied: {p['clahe']['applied']}   color correction: {p['color_correction']}")
    for a, (t, im) in zip(ax[i], [("input", lo), ("retinex stage", info["retinex"]),
                                  ("final", out), ("ground truth", gt)]):
        a.imshow(im); a.set_title(f"{r['low']}: {t}"); a.axis("off")
plt.tight_layout()
plt.savefig("results/failure_cases.png", dpi=110)
plt.show()