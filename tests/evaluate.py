import os, re, sys, csv
import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from classical.preprocessing import load_image, estimate_noise
from classical.classical_pipeline import enhance_classical

# usage: python -m tests.evaluate <low_dir> <high_dir> <tag> [limit]
# Pairs files by the number in their name (low00690.png <-> normal00690.png, 1.png <-> 1.png).
# Metrics: uint8 RGB, data_range=255, no brightness matching, mean over images.
low_dir, high_dir, tag = sys.argv[1], sys.argv[2], sys.argv[3]
limit = int(sys.argv[4]) if len(sys.argv) > 4 else None


def listing(d):
    return sorted(f for f in os.listdir(d)
                  if f.lower().endswith((".png", ".jpg", ".jpeg")) and not f.startswith("._"))


def key(name):
    m = re.findall(r"\d+", os.path.splitext(name)[0])
    return int(m[-1]) if m else None


highs = {key(f): f for f in listing(high_dir)}
pairs = [(f, highs[key(f)]) for f in listing(low_dir) if key(f) in highs]
skipped = len(listing(low_dir)) - len(pairs)
pairs = pairs[:limit]
print(f"{tag}: {len(pairs)} image pairs" + (f" ({skipped} low images had no match and were skipped)" if skipped else ""))
if not pairs:
    sys.exit("no pairs found. Check the two folder paths.")

CONFIGS = {
    "input (no processing)": None,
    "FULL pipeline":         {},
    "- adaptive Retinex":    {"use_adaptive": False},
    "- color correction":    {"use_color_correction": False},
    "- CLAHE":               {"use_clahe": False},
    "- denoise":             {"use_denoise": False},
    "- Retinex":             {"use_retinex": False},
    "Retinex only":          {"use_adaptive": False, "use_denoise": False,
                              "use_clahe": False, "use_color_correction": False},
}


def grad(x):
    g = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))


def edge_corr(a, b):
    return float(np.corrcoef(grad(a).ravel(), grad(b).ravel())[0, 1])


res = {k: [] for k in CONFIGS}      # per image: (psnr, ssim, ms, noise, edge_corr)
for i, (ln, hn) in enumerate(pairs):
    lo = load_image(os.path.join(low_dir, ln))
    gt = load_image(os.path.join(high_dir, hn))
    for k, cfg in CONFIGS.items():
        if cfg is None:
            out, ms = lo, 0.0
        else:
            out, info = enhance_classical(lo, cfg)
            ms = info["meta"]["latency_ms"]["total"]
        g = gt if gt.shape == out.shape else cv2.resize(gt, (out.shape[1], out.shape[0]),
                                                         interpolation=cv2.INTER_AREA)
        res[k].append((psnr(g, out, data_range=255),
                       ssim(g, out, channel_axis=2, data_range=255), ms,
                       estimate_noise(cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)), edge_corr(out, g)))
    if (i + 1) % 10 == 0:
        print(f"  {i + 1}/{len(pairs)}")

full = np.array(res["FULL pipeline"])
lines = ["| config | PSNR | SSIM | latency ms | noise | edge corr | dPSNR vs FULL |",
         "|---|---|---|---|---|---|---|"]
for k, v in res.items():
    m = np.array(v).mean(0)
    lines.append(f"| {k} | {m[0]:.2f} | {m[1]:.3f} | {m[2]:.0f} | {m[3]:.2f} | {m[4]:.3f} | {m[0] - full[:, 0].mean():+.2f} |")
table = "\n".join(lines)
print("\n" + table)

print(f"\nlatency FULL: mean {full[:, 2].mean():.0f} ms, p95 {np.percentile(full[:, 2], 95):.0f} ms")
inp = np.array(res["input (no processing)"])
d = full[:, 0] - inp[:, 0]
print(f"FULL beats raw input on PSNR for {int((d > 0).sum())}/{len(d)} images")
worst = np.argsort(full[:, 0])[:3]
print("lowest-PSNR images (failure-case candidates):",
      [(pairs[j][0], round(float(full[j, 0]), 2)) for j in worst])

os.makedirs("results", exist_ok=True)
with open(f"results/eval_{tag}.md", "w") as f:
    f.write(f"# classical pipeline, {tag}, {len(pairs)} images\n\n{table}\n")
with open(f"results/eval_{tag}_per_image.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["low", "high", "psnr", "ssim", "ms", "noise", "edge_corr"])
    for j, (ln, hn) in enumerate(pairs):
        w.writerow([ln, hn] + [round(float(x), 4) for x in full[j]])
print(f"\nsaved results\\eval_{tag}.md and results\\eval_{tag}_per_image.csv")