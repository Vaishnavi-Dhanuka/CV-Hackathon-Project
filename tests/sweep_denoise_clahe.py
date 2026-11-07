import os, sys, csv
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from classical.preprocessing import load_image
from classical.illumination import illumination_analysis
from classical.retinex import retinex_enhance
from classical.denoising import denoise
from classical.clahe import apply_clahe

root = sys.argv[1]                                   # path to our485
step = int(sys.argv[2]) if len(sys.argv) > 2 else 5  # every Nth image
low_dir, high_dir = os.path.join(root, "low"), os.path.join(root, "high")
names = sorted(f for f in os.listdir(low_dir)
               if f.lower().endswith(".png") and not f.startswith("._"))[::step]
print(f"{len(names)} images")

denoisers = [("none", "none", 0)] + [(f"{m}-{s}", m, s)
                                     for m in ("guided", "bilateral") for s in (8, 12, 18, 25)]
clahes = [("off", None, None), ("clip1.0", 1.0, 1.0),
          ("clip2.0 s0.5", 2.0, 0.5), ("clip2.0", 2.0, 1.0)]

scores = {}
for i, n in enumerate(names):
    lo = load_image(os.path.join(low_dir, n))
    gt = load_image(os.path.join(high_dir, n))
    L = illumination_analysis(lo)["illumination_map"]
    base = retinex_enhance(lo, method="guided", gamma=0.5, max_gain=8.0,
                           target_mean=0.5, strength=0.9, L=L)
    for dlabel, dm, ds in denoisers:
        dn = denoise(base, dm, ds)
        for clabel, clip, st in clahes:
            out = dn if clip is None else apply_clahe(dn, clip, (8, 8), st)
            scores.setdefault((dlabel, clabel), []).append(
                (psnr(gt, out), ssim(gt, out, channel_axis=2)))
    if (i + 1) % 10 == 0:
        print(f"  {i + 1}/{len(names)}")

rows = [(d, c, np.mean([s[0] for s in v]), np.mean([s[1] for s in v]))
        for (d, c), v in scores.items()]
rows.sort(key=lambda r: -r[2])
print(f"\n{'denoise':14s} {'clahe':14s} {'PSNR':>6} {'SSIM':>6}")
for d, c, p, s in rows:
    print(f"{d:14s} {c:14s} {p:6.2f} {s:6.3f}")

os.makedirs("results", exist_ok=True)
with open("results/sweep_denoise_clahe.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["denoise", "clahe", "psnr", "ssim"]); w.writerows(rows)