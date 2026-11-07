import cv2
import numpy as np


def apply_clahe(image, clip_limit=2.0, tile_grid=(8, 8), strength=1.0):
    """CLAHE on the L channel of LAB (colours untouched). uint8 RGB in, uint8 RGB out.
    clip_limit: higher = stronger contrast (and noise). tile_grid: smaller tiles = more local.
    strength:   0..1 blend with the input, the over-enhancement guard."""
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tuple(int(t) for t in tile_grid))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    if strength < 1.0:
        out = cv2.addWeighted(out, strength, image, 1.0 - strength, 0)
    return out


def sweep(image, clips=(1.0, 2.0, 3.0, 4.0), grids=((8, 8),)):
    """Run several settings; returns {label: image} for side-by-side experiments."""
    return {f"clip={c} grid={g[0]}x{g[1]}": apply_clahe(image, c, g) for g in grids for c in clips}