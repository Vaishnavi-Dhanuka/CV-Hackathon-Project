
from pathlib import Path
import shutil
import random

ROOT = Path(__file__).resolve().parents[1]

MODEL_A = ROOT / "data/model_datasets/Model_A"
MODEL_B = ROOT / "data/model_datasets/Model_B"

SYNTHETIC = (
    ROOT / "data/raw/LOL-v2/LOL-v2/LOL-v2/Synthetic/Train"
)

# Rebuild only Model B
if MODEL_B.exists():
    shutil.rmtree(MODEL_B)

for split in ("Train", "Val"):
    for kind in ("Low", "Normal"):
        (MODEL_B / split / kind).mkdir(parents=True, exist_ok=True)

# Copy the exact Real split used by Model A
for split in ("Train", "Val"):
    for kind in ("Low", "Normal"):
        source = MODEL_A / split / kind
        destination = MODEL_B / split / kind

        for file in source.glob("*.png"):
            shutil.copy2(file, destination / file.name)

# Prepare deterministic Synthetic split
low_files = sorted((SYNTHETIC / "Low").glob("*.png"))

rng = random.Random(42)
rng.shuffle(low_files)

val_files = set(f.name for f in low_files[:90])
train_files = set(f.name for f in low_files[90:])

for filename in sorted(train_files | val_files):
    split = "Val" if filename in val_files else "Train"

    low_src = SYNTHETIC / "Low" / filename
    normal_src = SYNTHETIC / "Normal" / filename

    if not normal_src.exists():
        raise FileNotFoundError(normal_src)

    prefix = "synthetic_"

    shutil.copy2(
        low_src,
        MODEL_B / split / "Low" / (prefix + filename)
    )
    shutil.copy2(
        normal_src,
        MODEL_B / split / "Normal" / (prefix + filename)
    )

for split in ("Train", "Val"):
    low_count = len(list((MODEL_B / split / "Low").glob("*.png")))
    normal_count = len(list((MODEL_B / split / "Normal").glob("*.png")))
    print(f"Model B {split}: Low={low_count}, Normal={normal_count}")

print("Model B rebuilt with shared Real validation images.")
