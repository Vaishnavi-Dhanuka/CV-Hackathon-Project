import os, sys, csv
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from classical.preprocessing import load_image, compute_characteristics
from classical.illumination import illumination_analysis
from classical.classical_pipeline import enhance_classical
from classical.adaptive import FEATURES

# usage: python -m tests.sweep_adaptive <path to our485>
root = sys.argv[1]
low_dir, high_dir = os.path.join(root, "low"), os.path.join(root, "high")
names_all = sorted(f for f in os.listdir(low_dir)
                   if f.lower().endswith(".png") and not f.startswith("._"))
fit_names = names_all[0::5]      # used to choose the rule
chk_names = names_all[2::5]      # different images, only used to verify it
print(f"fit {len(fit_names)} images, check {len(chk_names)} images")

GRID = [(tm, g) for tm in (0.4, 0.5, 0.6) for g in (0.4, 0.5, 0.7)]


def collect(names, tag):
    rows = []
    for i, n in enumerate(names):
        lo = load_image(os.path.join(low_dir, n))
        gt = load_image(os.path.join(high_dir, n))
        stats, illum = compute_characteristics(lo), illumination_analysis(lo)
        feats = {k: f(stats, illum) for k, f in FEATURES.items()}
        sc = []
        for tm, g in GRID:
            out, _ = enhance_classical(lo, {"retinex_override": dict(target_mean=tm, gamma=g)})
            sc.append(psnr(gt, out))
        rows.append({"name": n, "feats": feats, "scores": sc})
        if (i + 1) % 20 == 0:
            print(f"  {tag} {i + 1}/{len(names)}")
    return rows


fit, chk = collect(fit_names, "fit"), collect(chk_names, "check")
Sf, Sc = np.array([r["scores"] for r in fit]), np.array([r["scores"] for r in chk])

print(f"\n{'target_mean':>11} {'gamma':>6} {'fit PSNR':>9} {'check PSNR':>11}")
for j, (tm, g) in enumerate(GRID):
    print(f"{tm:11.1f} {g:6.1f} {Sf[:, j].mean():9.2f} {Sc[:, j].mean():11.2f}")
bi = int(np.argmax(Sf.mean(0)))
print(f"\nbest FIXED setting on fit set: target_mean={GRID[bi][0]}, gamma={GRID[bi][1]}")
print(f"oracle (best setting per image, upper bound): fit {Sf.max(1).mean():.2f}, check {Sc.max(1).mean():.2f}")

results = {}
print(f"\n{'feature':16s} {'fit PSNR':>9} {'check PSNR':>11} {'check gain vs fixed':>20}")
for feat in FEATURES:
    vf = np.array([r["feats"][feat] for r in fit])
    vc = np.array([r["feats"][feat] for r in chk])
    edges = np.percentile(vf, [100 / 3, 200 / 3])
    bf, bc = np.digitize(vf, edges), np.digitize(vc, edges)
    choice = [int(np.argmax(Sf[bf == b].mean(0))) if (bf == b).any() else bi for b in range(3)]
    fs = np.mean([Sf[j, choice[bf[j]]] for j in range(len(fit))])
    cs = np.mean([Sc[j, choice[bc[j]]] for j in range(len(chk))])
    results[feat] = (fs, cs, edges, choice, bc)
    print(f"{feat:16s} {fs:9.2f} {cs:11.2f} {cs - Sc[:, bi].mean():+20.2f}")

best = max(results, key=lambda k: results[k][0])      # chosen on the FIT set only
fs, cs, edges, choice, bc = results[best]
print(f"\nchosen feature (best on fit set): {best}")
print(f"bin edges: {edges[0]:.4f}, {edges[1]:.4f}")
for b, c in enumerate(choice):
    print(f"  bin {b}: target_mean={GRID[c][0]}, gamma={GRID[c][1]}")

# SSIM check on the held-out images: adaptive rule vs best fixed
s_fix, s_ada = [], []
for j, n in enumerate(chk_names):
    lo = load_image(os.path.join(low_dir, n))
    gt = load_image(os.path.join(high_dir, n))
    tm, g = GRID[bi]
    a, _ = enhance_classical(lo, {"retinex_override": dict(target_mean=tm, gamma=g)})
    tm, g = GRID[choice[bc[j]]]
    b_, _ = enhance_classical(lo, {"retinex_override": dict(target_mean=tm, gamma=g)})
    s_fix.append(ssim(gt, a, channel_axis=2)); s_ada.append(ssim(gt, b_, channel_axis=2))
gain = cs - Sc[:, bi].mean()
print(f"\nCHECK SET: fixed {Sc[:, bi].mean():.2f} dB / {np.mean(s_fix):.3f}  vs  adaptive {cs:.2f} dB / {np.mean(s_ada):.3f}  (gain {gain:+.2f} dB)")

print("\n=== paste into classical\\adaptive.py, replacing 'ADAPTIVE_RULE = None' ===")
if gain >= 0.10:
    print("ADAPTIVE_RULE = {")
    print(f'    "feature": "{best}",')
    print(f'    "edges": ({edges[0]:.4f}, {edges[1]:.4f}),')
    print('    "params": [' + ", ".join(f"dict(target_mean={GRID[c][0]}, gamma={GRID[c][1]})" for c in choice) + "],")
    print("}")
else:
    print(f"# gain on the check set is only {gain:+.2f} dB: keep ADAPTIVE_RULE = None (fixed target_mean={GRID[bi][0]}, gamma={GRID[bi][1]})")
    print(f"# and if that setting is not 0.5/0.5, change RETINEX_PARAMS in classical_pipeline.py to it")

os.makedirs("results", exist_ok=True)
with open("results/sweep_adaptive.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["split", "name"] + list(FEATURES) + [f"tm{t}_g{g}" for t, g in GRID])
    for tag, rows in (("fit", fit), ("check", chk)):
        for r in rows:
            w.writerow([tag, r["name"]] + [round(r["feats"][k], 4) for k in FEATURES] + [round(s, 3) for s in r["scores"]])