from pathlib import Path
from PIL import Image
import csv
import time
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deep_learning"))

from retinex_inference import enhance_image

DATA = ROOT / "data/raw/ExDark/ExDark/ExDark"
OUT = ROOT / "results/evaluation_outputs/ExDark_Retinexformer"
CSV = ROOT / "results/metrics/ExDark_generalization.csv"

rows = []

for category in sorted(p for p in DATA.iterdir() if p.is_dir()):
    images = sorted(
        p for p in category.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )[:3]

    for path in images:
        try:
            with Image.open(path) as image:
                start = time.perf_counter()
                enhanced = enhance_image(image)
                latency = (time.perf_counter() - start) * 1000

            destination = OUT / category.name / f"{path.stem}.png"
            destination.parent.mkdir(parents=True, exist_ok=True)
            enhanced.save(destination)

            rows.append({
                "category": category.name,
                "filename": path.name,
                "latency_ms": round(latency, 2),
                "output": str(destination.relative_to(ROOT))
            })
            print(f"OK: {category.name}/{path.name} ({latency:.1f} ms)")

        except Exception as exc:
            print(f"FAILED: {path} — {exc}")

CSV.parent.mkdir(parents=True, exist_ok=True)
with CSV.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f, fieldnames=["category", "filename", "latency_ms", "output"]
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"\nSuccessful images: {len(rows)}")
print(f"Average latency: {sum(r['latency_ms'] for r in rows)/len(rows):.2f} ms" if rows else "No successful images")
print(f"CSV: {CSV}")
print(f"Outputs: {OUT}")
