import cv2
import numpy as np

L_MIN = 0.02   # floor for illumination: avoids dividing by ~0


def estimate_illumination(image, radius=None, eps=1e-4):
    """Illumination map L in [L_MIN, 1], float32, shape (H, W).
    radius: guided-filter window (default ~4% of the short side).
    eps:    edge-vs-smoothness knob. Small = follows edges closely.
    """
    h, w = image.shape[:2]
    if radius is None:
        radius = max(8, int(0.04 * min(h, w)))
    img = image.astype(np.float32) / 255.0
    l0 = img.max(axis=2)                                   # initial estimate
    guide = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    L = cv2.ximgproc.guidedFilter(guide, l0, radius, eps)  # edge-aware smoothing
    return np.clip(L, L_MIN, 1.0).astype(np.float32)


def decompose(image, radius=None, eps=1e-4):
    """Retinex decomposition I = R * L. Returns (L, R), both float32; R in [0,1]."""
    img = image.astype(np.float32) / 255.0
    L = estimate_illumination(image, radius, eps)
    R = np.clip(img / L[:, :, None], 0.0, 1.0)
    return L, R


def analyze_regions(L, dark_thresh=0.2):
    """Find dark and unevenly lit regions from the illumination map."""
    med = float(np.median(L))
    dark_mask = L < dark_thresh
    uneven_mask = (L > 2.0 * med) | (L < 0.5 * med)   # far from the typical light level
    return {
        "dark_mask": dark_mask,
        "uneven_mask": uneven_mask,
        "illum_mean": float(L.mean()),
        "illum_cv": float(L.std() / (L.mean() + 1e-6)),   # unevenness score
        "dark_region_fraction": float(dark_mask.mean()),
        "uneven_fraction": float(uneven_mask.mean()),
    }


def illumination_analysis(image, radius=None, eps=1e-4):
    """Main entry point. Returns a dict:
    illumination_map (float, for retinex.py), illumination / reflectance
    (uint8 RGB, for display), masks, and scores."""
    L, R = decompose(image, radius, eps)
    info = analyze_regions(L)

    lo, hi = np.percentile(L, [1, 99])                    # stretch so dark maps are visible
    L_vis = np.clip((L - lo) / (hi - lo + 1e-6), 0, 1)
    out = {
        "illumination_map": L,
        "illumination": cv2.cvtColor((L_vis * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB),
        "reflectance": (R * 255 + 0.5).astype(np.uint8),
    }
    out.update(info)
    return out


def visualize(image, out, save_path=None):
    import matplotlib.pyplot as plt
    panels = [
        (image, "original"),
        (out["illumination"], "illumination (stretched)"),
        (out["reflectance"], "reflectance"),
        (out["dark_mask"], "dark regions"),
        (out["uneven_mask"], "uneven regions"),
    ]
    fig, ax = plt.subplots(1, 5, figsize=(20, 3.6))
    for a, (im, title) in zip(ax, panels):
        if im.ndim == 2:
            a.imshow(im, cmap="gray", vmin=0, vmax=1)
        else:
            a.imshow(im)
        a.set_title(title)
        a.axis("off")
    plt.tight_layout()
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=120)
    plt.show()