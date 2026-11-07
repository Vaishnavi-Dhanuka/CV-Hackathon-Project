
from pathlib import Path
import random
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
RANDOM_SEED = 42
VAL_FRACTION = 0.10

REAL = ROOT / "data/raw/Kaggle/archive/LOL-v2/Real_captured"
SYNTHETIC = ROOT / "data/raw/LOL-v2/LOL-v2/LOL-v2/Synthetic"
OUTPUT = ROOT / "data/model_datasets"

def collect_pairs(source, prefix, real=False):
    pairs = []
    low_dir = source / "Train/Low"
    gt_dir = source / "Train/Normal"

    for low in sorted(low_dir.glob("*.png")):
        if real:
            match = re.search(r"(\d+)$", low.stem)
            if not match:
                continue
            gt = gt_dir / f"normal{match.group(1)}{low.suffix}"
        else:
            gt = gt_dir / low.name

        if not gt.exists():
            raise FileNotFoundError(f"Missing GT for {low}: {gt}")

        pairs.append((low, gt, f"{prefix}_{low.name}"))

    return pairs

def prepare(name, pairs):
    rng = random.Random(RANDOM_SEED)
    pairs = list(pairs)
    rng.shuffle(pairs)

    val_count = max(1, round(len(pairs) * VAL_FRACTION))
    val_pairs = pairs[:val_count]
    train_pairs = pairs[val_count:]

    for split, split_pairs in [
        ("Train", train_pairs),
        ("Val", val_pairs),
    ]:
        low_out = OUTPUT / name / split / "Low"
        gt_out = OUTPUT / name / split / "Normal"
        low_out.mkdir(parents=True, exist_ok=True)
        gt_out.mkdir(parents=True, exist_ok=True)

        for low, gt, filename in split_pairs:
            shutil.copy2(low, low_out / filename)
            shutil.copy2(gt, gt_out / filename)

    print(
        f"{name}: train={len(train_pairs)}, "
        f"validation={len(val_pairs)}, total={len(pairs)}"
    )

real_pairs = collect_pairs(REAL, "real", real=True)
prepare("Model_A", real_pairs)

synthetic_pairs = collect_pairs(SYNTHETIC, "synthetic")
prepare("Model_B", real_pairs + synthetic_pairs)

print("Dataset preparation complete.")
print("Official test folders were not copied or modified.")
