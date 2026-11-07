from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

root = Path(__file__).resolve().parents[1]

folders = [
    ("Low-light input", root / "data/raw/Kaggle/archive/LOL-v2/Real_captured/Test/Low"),
    ("Retinexformer", root / "results/evaluation_outputs/Retinexformer"),
    ("Model B", root / "results/metrics/model_B/official_test"),
    ("Ground truth", root / "data/raw/Kaggle/archive/LOL-v2/Real_captured/Test/Normal"),
]

names = ["low00690.png", "low00712.png", "low00763.png", "low00789.png"]
out = root / "results/metrics/final_model_comparison.png"

w, h, label_h, gap, margin = 360, 240, 38, 14, 24
sheet = Image.new("RGB", (margin*2 + 4*w + 3*gap,
                          margin*2 + 4*(h+label_h) + 3*gap), "#20242b")
draw = ImageDraw.Draw(sheet)

try:
    font = ImageFont.truetype("arial.ttf", 19)
except OSError:
    font = ImageFont.load_default()

for r, name in enumerate(names):
    for c, (label, folder) in enumerate(folders):
        filename = name.replace("low", "normal", 1) if label == "Ground truth" else name
        path = folder / filename
        x = margin + c*(w+gap)
        y = margin + r*(h+label_h+gap)
        draw.text((x, y), label, fill="white", font=font)

        if not path.exists():
            raise FileNotFoundError(path)

        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((w, h), Image.Resampling.LANCZOS)
            tile = Image.new("RGB", (w, h), "black")
            tile.paste(im, ((w-im.width)//2, (h-im.height)//2))
            sheet.paste(tile, (x, y+label_h))

out.parent.mkdir(parents=True, exist_ok=True)
sheet.save(out)
print(f"Saved: {out}")
