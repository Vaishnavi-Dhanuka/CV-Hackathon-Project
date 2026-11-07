import sys
import numpy as np
import matplotlib.pyplot as plt
from classical.preprocessing import prepare_image

path = sys.argv[1]
img, s = prepare_image(path)
print("shape", img.shape, img.dtype)
for k, v in s.items():
    if k != "histogram":
        print(f"{k:20s}", np.round(v, 3) if not isinstance(v, tuple) else v)

# --- bad input checks ---
tests = {
    "None": None,
    "grayscale": np.random.randint(0, 255, (100, 100), np.uint8),
    "RGBA": np.random.randint(0, 255, (100, 100, 4), np.uint8),
    "float 0-1": np.random.rand(100, 100, 3),
    "16-bit": np.random.randint(0, 65535, (100, 100, 3)).astype(np.uint16),
    "tiny 4x4": np.zeros((4, 4, 3), np.uint8),
    "huge 3000x4000": np.zeros((3000, 4000, 3), np.uint8),
    "missing file": "data/nope.png",
}
for name, x in tests.items():
    try:
        out, _ = prepare_image(x)
        print(f"{name:15s} OK   -> {out.shape} {out.dtype}")
    except ValueError as e:
        print(f"{name:15s} REJECTED: {e}")

fig, ax = plt.subplots(1, 3, figsize=(14, 3.5))
ax[0].imshow(img); ax[0].set_title("input"); ax[0].axis("off")
ax[1].plot(s["histogram"]); ax[1].set_title("gray histogram")
ax[2].bar(["R", "G", "B"], s["channel_mean"], color=["r", "g", "b"]); ax[2].set_title("channel means")
plt.tight_layout(); plt.show()