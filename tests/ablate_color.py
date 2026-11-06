import os, sys
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from classical.preprocessing import load_image
from classical.classical_pipeline import enhance_classical
from classical.color_correction import color_correct, choose_color_params

# usage: python -m tests.ablate_color <path to our485> [step]
root = sys.argv[1]
step = int(sys.argv[2]) if len(sys.argv) > 2 else 10
low_dir, high_dir = os.path.join(root, "low"), os.path.join(root, "high")
names = sorted(f for f in os.listdir(low_dir)
               if f.lower().endswith(".png") and not f.startswith("._"))[::step]

NO_HL = 0.999   # knee so high that highlight protection does nothing
variants = {
    "none (cc off)": None,
    "wb only":       lambda p: dict(wb_strength=p["wb_strength"], gamma=1.0, contrast_strength=0.0, knee=NO_HL),
    "gamma only":    lambda p: dict(wb_strength=0.0, gamma=p["gamma"], contrast_strength=0.0, knee=NO_HL),
    "contrast only": lambda p: dict(wb_strength=0.0, gamma=1.0, contrast_strength=0.5, knee=NO_HL),
    "highlight only": lambda p: dict(wb_strength=0.0, gamma=1.0, contrast_strength=0.0, knee=p["knee"]),
    "all four":      lambda p: p,
}

res = {k: [] for k in variants}      # (psnr, ssim) per image
extra = []                           # per-image params/measurements
for n in names:
    lo = load_image(os.path.join(low_dir, n))
    gt = load_image(os.path.join(high_dir, n))
    base, _ = enhance_classical(lo, {"use_color_correction": False})
    p, meas = choose_color_params(base)
    extra.append((n, p, meas))
    for k, fn in variants.items():
        out = base if fn is None else color_correct(base, **fn(p))
        res[k].append((psnr(gt, out), ssim(gt, out, channel_axis=2)))

off = np.array([a for a, _ in res["none (cc off)"]])
print(f"{len(names)} images\n")
print(f"{'variant':16s} {'PSNR':>6} {'SSIM':>6} {'helps':>7} {'mean dB':>8} {'worst dB':>9}  worst image")
for k, v in res.items():
    ps = np.array([a for a, _ in v]); ss = np.array([b for _, b in v])
    d = ps - off
    w = int(np.argmin(d))
    print(f"{k:16s} {ps.mean():6.2f} {ss.mean():6.3f} {int((d > 0).sum()):>3}/{len(d):<3} "
          f"{d.mean():+8.2f} {d.min():+9.2f}  {names[w]}")

# the single worst case for 'all four' (use it as the viva failure case)
d_all = np.array([a for a, _ in res["all four"]]) - off
w = int(np.argmin(d_all))
n, p, meas = extra[w]
print(f"\nWORST CASE for 'all four': {n} ({d_all[w]:+.2f} dB)")
print("params  :", {k: round(v, 3) for k, v in p.items()})
print("measured:", {k: round(v, 3) for k, v in meas.items()})

# does the white balance hurt more when the measured cast is high?
cast = np.array([m["color_cast"] for _, _, m in extra])
d_wb = np.array([a for a, _ in res["wb only"]]) - off
print(f"\ncorr(measured cast, wb-only dB change) = {np.corrcoef(cast, d_wb)[0, 1]:+.2f}")