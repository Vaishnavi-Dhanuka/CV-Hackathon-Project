from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
LOW = ROOT / "data/raw/Kaggle/archive/LOL-v2/Real_captured/Test/Low"
GT = ROOT / "data/raw/Kaggle/archive/LOL-v2/Real_captured/Test/Normal"
RET = ROOT / "results/evaluation_outputs/Retinexformer"
HYB = ROOT / "results/evaluation_outputs/Hybrid_Adaptive"
OUT = ROOT / "results/metrics/labelled_comparison.png"

names = ["low00712.png", "low00713.png", "low00763.png", "low00764.png"]
columns = [
    ("Input (low-light)", LOW),
    ("Ground truth", GT),
    ("Retinexformer", RET),
    ("Adaptive hybrid", HYB),
]
thumb_w, thumb_h, label_h, gap = 360, 240, 34, 12
margin = 20
rows = len(names)
canvas_w = margin * 2 + len(columns) * thumb_w + (len(columns) - 1) * gap
canvas_h = margin * 2 + rows * (thumb_h + label_h) + (rows - 1) * gap
sheet = Image.new("RGB", (canvas_w, canvas_h), (30, 30, 30))
draw = ImageDraw.Draw(sheet)
try:
    font = ImageFont.truetype("arial.ttf", 18)
except OSError:
    font = ImageFont.load_default()

for r, low_name in enumerate(names):
    for c, (label, folder) in enumerate(columns):
        filename = low_name.replace("low", "normal", 1) if label == "Ground truth" else low_name
        path = folder / filename
        x = margin + c * (thumb_w + gap)
        y = margin + r * (thumb_h + label_h + gap)
        draw.text((x, y), f"{label} — {low_name}", fill="white", font=font)
        if not path.exists():
            draw.rectangle((x, y + label_h, x + thumb_w, y + label_h + thumb_h), fill=(90, 30, 30))
            draw.text((x + 10, y + label_h + 10), f"Missing: {path}", fill="white", font=font)
            continue
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
            tile = Image.new("RGB", (thumb_w, thumb_h), (0, 0, 0))
            tile.paste(im, ((thumb_w - im.width) // 2, (thumb_h - im.height) // 2))
            sheet.paste(tile, (x, y + label_h))

OUT.parent.mkdir(parents=True, exist_ok=True)
sheet.save(OUT)
print(f"Saved comparison sheet: {OUT}")
