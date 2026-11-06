import time
from contextlib import contextmanager

import cv2
import numpy as np

from classical.preprocessing import to_rgb_uint8, resize_if_large, compute_characteristics
from classical.illumination import illumination_analysis
from classical.retinex import retinex_enhance
from classical.clahe import apply_clahe
from classical.denoising import denoise

DEFAULT_CONFIG = {
    "use_retinex": True,
    "use_clahe": True,
    "use_denoise": True,
    "use_color_correction": True,
}

# Fixed defaults for now; adaptive selection (Task 7) will replace these with per-image values.
RETINEX_PARAMS = dict(method="guided", gamma=0.5, max_gain=8.0, target_mean=0.5, strength=0.9)


@contextmanager
def _timer(store, name):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        store[name] = round((time.perf_counter() - t0) * 1000.0, 2)   # milliseconds


def _clean_stats(s):
    """Make stats JSON-friendly: drop histogram, arrays -> lists, numbers -> rounded floats."""
    out = {}
    for k, v in s.items():
        if k == "histogram":
            continue
        if isinstance(v, np.ndarray):
            out[k] = [round(float(x), 3) for x in v]
        elif isinstance(v, (tuple, list)):
            out[k] = [int(x) for x in v]
        else:
            out[k] = round(float(v), 4)
    return out


def enhance_classical(image, config=None):
    """Input: uint8 RGB image (any size). Output: (enhanced uint8 RGB, intermediate_outputs).
    intermediate_outputs: six uint8 RGB images (same size as the returned image) + "meta" dict."""
    t_total = time.perf_counter()
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    lat = {}

    with _timer(lat, "preprocess"):
        img = to_rgb_uint8(image)                    # raises ValueError on bad input
        orig_size = img.shape[:2]
        img, scale = resize_if_large(img)
        stats = compute_characteristics(img)         # measured on the image the pipeline sees

    with _timer(lat, "illumination"):
        illum = illumination_analysis(img)

    out = img
    with _timer(lat, "retinex"):
        if cfg["use_retinex"]:
            out = retinex_enhance(out, L=illum["illumination_map"], **RETINEX_PARAMS)
    retinex_out = out

    with _timer(lat, "denoise"):
        denoised = denoise(out, "guided", sigma=12) if cfg["use_denoise"] else out
    with _timer(lat, "clahe"):
        clahe_out = apply_clahe(denoised, 2.0, (8, 8)) if cfg["use_clahe"] else denoised
    with _timer(lat, "color_correction"):
        final = clahe_out                            # TODO: color_correction.py

    lat["total"] = round((time.perf_counter() - t_total) * 1000.0, 2)

    in_stats = _clean_stats(stats)
    in_stats["original_size"] = [int(orig_size[0]), int(orig_size[1])]
    in_stats["scale"] = round(float(scale), 4)

    meta = {
        "latency_ms": lat,
        "config": {
            "flags": cfg,
            "params": {"retinex": dict(RETINEX_PARAMS) if cfg["use_retinex"] else None},
            "adaptive": False,       # becomes True once Task 7 is in
        },
        "stages_implemented": {"retinex": True, "clahe": True, "denoise": True,
                       "color_correction": False},
        "input_stats": in_stats,
        "illumination_stats": {k: round(float(illum[k]), 4) for k in
                               ["illum_mean", "illum_cv", "dark_region_fraction", "uneven_fraction"]},
        "output_size": [int(final.shape[0]), int(final.shape[1])],
        "device": {"type": "cpu", "backend": "OpenCV/NumPy",
                   "opencv": cv2.__version__, "cv_threads": int(cv2.getNumThreads())},
    }

    intermediate = {
        "illumination": illum["illumination"],
        "reflectance": illum["reflectance"],
        "retinex": retinex_out,
        "clahe": clahe_out,
        "denoised": denoised,
        "final": final,
        "meta": meta,
    }
    return final, intermediate