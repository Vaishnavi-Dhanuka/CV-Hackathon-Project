import numpy as np
from classical.classical_pipeline import enhance_classical

rng = np.random.default_rng(0)
cases = {
    "all black": np.zeros((300, 400, 3), np.uint8),
    "all white": np.full((300, 400, 3), 255, np.uint8),
    "tiny 8x8": rng.integers(0, 255, (8, 8, 3), dtype=np.uint8),
    "grayscale 2D": rng.integers(0, 60, (300, 400), dtype=np.uint8),
    "RGBA 4-channel": rng.integers(0, 60, (300, 400, 4), dtype=np.uint8),
    "float image": rng.random((300, 400, 3)).astype(np.float32) * 0.2,
    "huge 3000x4000": rng.integers(0, 40, (3000, 4000, 3), dtype=np.uint8),
    "None": None,
}
for name, img in cases.items():
    try:
        out, inter = enhance_classical(img)
        print(f"OK    {name}: out {out.shape} {out.dtype}")
    except Exception as e:
        print(f"FAIL  {name}: {type(e).__name__}: {e}")