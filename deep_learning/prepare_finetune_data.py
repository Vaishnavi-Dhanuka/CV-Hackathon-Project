
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]

SOURCE = ROOT / "data/raw/Kaggle/archive/LOL-v2/Real_captured"
DEST = ROOT / "data/finetune"

for split in ("Train", "Test"):
    low_dir = SOURCE / split / "Low"
    gt_dir = SOURCE / split / "Normal"

    out_low = DEST / split / "Low"
    out_gt = DEST / split / "Normal"

    out_low.mkdir(parents=True, exist_ok=True)
    out_gt.mkdir(parents=True, exist_ok=True)

    for low_path in sorted(low_dir.glob("*")):
        if not low_path.is_file():
            continue

        match = re.search(r"(\d+)$", low_path.stem)
        if not match:
            continue

        gt_path = gt_dir / f"normal{match.group(1)}{low_path.suffix}"

        if not gt_path.exists():
            raise FileNotFoundError(gt_path)

        filename = f"{match.group(1)}{low_path.suffix}"
        shutil.copy2(low_path, out_low / filename)
        shutil.copy2(gt_path, out_gt / filename)

    print(f"{split}: {len(list(out_low.iterdir()))} paired images")

print("Dataset preparation complete.")
