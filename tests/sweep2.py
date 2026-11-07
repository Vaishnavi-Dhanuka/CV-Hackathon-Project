import os, sys, csv
import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from classical.preprocessing import load_image, estimate_noise
from classical.illumination import illumination_analysis
from classical.retinex import retinex_enhance
from classical.denoising import denoise
from classical.clahe import apply_clahe

root = sys.argv[1]
step = int(sys.argv[2]) if len(sys.argv) > 2 else 10
low_dir, high_dir = os.path.join(root, "low"), os.path.join(root, "high")
names = sorted(f for f in os.listdir(low_dir)
               if f.lower().endswith(".png") and not f.startswith("._"))[::step]
print(f"{len(names)} images")

def gray(x):
    return cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)

def contrast(x):
    p5, p95 = np.percentile(gray(x), [5, 95])
    return float(p95 - p5)

SIGMAS = (25, 35, 50, 70)
cfgs = [("none", "none", 0)] + [(f"{m}-{s}", m, s)
                                for m in ("guided", "bilateral") for s in SIGMAS]

# ---------- Part A: sigma sweep (CLAHE off) ----------
data, scores = [], {}
for i, n in enumerate(names):
    lo = load_image(os.path.join(low_dir, n))
    gt = load_image(os.path.join(high_dir, n))
    L = illumination_analysis(lo)["illumination_map"]
    base = retinex_enhance(lo, method="guided", gamma=0.5, max_gain=8.0,
                           target_mean=0.5, strength=0.9, L=L)
    rec = {"name": n, "base": base, "gt": gt,
           "noise": estimate_noise(gray(base)), "psnr": {}}
    for label, m, s in cfgs:
        dn = denoise(base, m, s)
        p, q = psnr(gt, dn), ssim(gt, dn, channel_axis=2)
        scores.setdefault(label, []).append((p, q))
        rec["psnr"][label] = p
    data.append(rec)
    if (i + 1) % 10 == 0:
        print(f"  {i + 1}/{len(names)}")

rows = sorted(((k, np.mean([a for a, _ in v]), np.mean([b for _, b in v]))
               for k, v in scores.items()), key=lambda r: -r[1])
print(f"\nPART A: denoise sweep, CLAHE off\n{'denoise':14s} {'PSNR':>6} {'SSIM':>6}")
for k, p, q in rows:
    print(f"{k:14s} {p:6.2f} {q:6.3f}")

best = rows[0][0]
bm, bs = next((m, s) for l, m, s in cfgs if l == best)
print(f"\nbest denoiser by PSNR: {best}")

# ---------- Part B: does CLAHE help low-contrast images? ----------
settings = [("clip1.0 s0.5", 1.0, 0.5), ("clip2.0 s0.5", 2.0, 0.5)]
con, deltas = [], {k: [] for k, _, _ in settings}
for rec in data:
    dn = denoise(rec["base"], bm, bs)
    off = psnr(rec["gt"], dn)
    con.append(contrast(dn))
    for k, clip, st in settings:
        out = apply_clahe(dn, clip, (8, 8), st)
        deltas[k].append(psnr(rec["gt"], out) - off)
con = np.array(con)
order = np.argsort(con)
groups = np.array_split(order, 3)
print("\nPART B: PSNR change from CLAHE (dB), grouped by contrast after denoising")
print(f"{'group':8s} {'contrast range':>16s} " + " ".join(f"{k:>14s}" for k, _, _ in settings))
for name, g in zip(("low", "mid", "high"), groups):
    rng = f"{con[g].min():.0f}-{con[g].max():.0f}"
    print(f"{name:8s} {rng:>16s} " +
          " ".join(f"{np.mean(np.array(deltas[k])[g]):+14.2f}" for k, _, _ in settings))
for k, _, _ in settings:
    d = np.array(deltas[k])
    print(f"{k}: helps {int((d > 0).sum())}/{len(d)} images, corr with contrast = "
          f"{np.corrcoef(con, d)[0, 1]:+.2f}")

# ---------- Part C: does best guided sigma follow measured noise? ----------
noise = np.array([r["noise"] for r in data])
bestsig = np.array([max(SIGMAS, key=lambda s: r["psnr"][f"guided-{s}"]) for r in data])
print("\nPART C: best guided sigma vs noise measured on the Retinex output")
print(f"{'group':8s} {'noise range':>14s} {'mean best sigma':>16s}")
for name, g in zip(("low", "mid", "high"), np.array_split(np.argsort(noise), 3)):
    print(f"{name:8s} {noise[g].min():6.1f}-{noise[g].max():<6.1f} {bestsig[g].mean():16.1f}")
print(f"corr(noise, best sigma) = {np.corrcoef(noise, bestsig)[0, 1]:+.2f}")
print("images per best sigma:", {int(s): int((bestsig == s).sum()) for s in SIGMAS})

os.makedirs("results", exist_ok=True)
with open("results/sweep2_per_image.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["name", "noise_after_retinex", "contrast_after_denoise", "best_guided_sigma"]
               + [f"delta_{k}" for k, _, _ in settings])
    for j, r in enumerate(data):
        w.writerow([r["name"], round(r["noise"], 2), round(con[j], 1), int(bestsig[j])]
                   + [round(deltas[k][j], 3) for k, _, _ in settings])