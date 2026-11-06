import os, sys, csv
from classical.preprocessing import load_image, save_image
from classical.classical_pipeline import enhance_classical

# usage: python -m tests.dump_outputs <input_dir> <output_dir>
# Saves enhanced images as PNG under the SAME base filename, plus <output_dir>\_run_log.csv
in_dir, out_dir = sys.argv[1], sys.argv[2]
os.makedirs(out_dir, exist_ok=True)

names = sorted(f for f in os.listdir(in_dir)
               if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")) and not f.startswith("._"))
print(f"{len(names)} images in {in_dir}")

rows, failed = [], []
for i, n in enumerate(names):
    try:
        img = load_image(os.path.join(in_dir, n))
        out, info = enhance_classical(img)
    except ValueError as e:
        failed.append((n, str(e)))
        print(f"  SKIPPED {n}: {e}")
        continue
    save_image(out, os.path.join(out_dir, os.path.splitext(n)[0] + ".png"))
    m = info["meta"]
    p = m["config"]["params"]
    rows.append([n, m["input_stats"]["original_size"][0], m["input_stats"]["original_size"][1],
                 m["output_size"][0], m["output_size"][1], m["latency_ms"]["total"],
                 (m["config"]["adaptive_rule"] or {}).get("bin"),
                 p["denoise"]["sigma"] if p["denoise"] else None,
                 p["clahe"]["applied"],
                 p["color_correction"]["wb_strength"] if p["color_correction"] else None])
    if (i + 1) % 20 == 0:
        print(f"  {i + 1}/{len(names)}")

with open(os.path.join(out_dir, "_run_log.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["file", "in_h", "in_w", "out_h", "out_w", "latency_ms",
                "retinex_bin", "denoise_sigma", "clahe_applied", "wb_strength"])
    w.writerows(rows)

print(f"\nsaved {len(rows)} images to {out_dir}; skipped {len(failed)}")
for n, e in failed:
    print("  skipped:", n, "-", e)