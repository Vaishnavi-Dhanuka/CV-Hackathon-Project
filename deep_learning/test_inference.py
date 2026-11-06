
from pathlib import Path
from retinex_inference import enhance_file

ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = ROOT / "data/raw/Kaggle/archive/LOL-v2/Real_captured/Test/Low"
OUTPUT_DIR = ROOT / "results/outputs/retinexformer_batch"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

images = sorted(INPUT_DIR.glob("*.png"))

for index, image_path in enumerate(images, 1):
    output_path = OUTPUT_DIR / image_path.name
    enhance_file(image_path, output_path)

    print(f"[{index}/{len(images)}] {image_path.name}")

print(f"\nProcessed {len(images)} images.")
print(f"Output directory: {OUTPUT_DIR}")
