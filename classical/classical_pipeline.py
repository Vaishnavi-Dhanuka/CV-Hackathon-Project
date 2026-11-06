import time
from contextlib import contextmanager

import cv2
import numpy as np

from classical.preprocessing import (to_rgb_uint8, resize_if_large,
                                     compute_characteristics, estimate_noise)
from classical.illumination import illumination_analysis
from classical.retinex import retinex_enhance
from classical.clahe import apply_clahe
from classical.denoising import denoise
from classical.color_correction import color_correct, choose_color_params
from classical.adaptive import choose_retinex_params

DEFAULT_CONFIG = {
    "use_retinex": True,
    "use_clahe": True,
    "use_denoise": True,
    "use_color_correction": True,
    "use_adaptive": True,        # extra flag: False = fixed Retinex params (for the ablation)
}

RETINEX_PARAMS = dict(method="guided", gamma=0.5, max_gain=8.0, target_mean=0.5, strength=0.9)
# Chosen by sweeps on a slice of LOL our485 (see sweep2.py), not on test data.
DENOISE_SIGMA_PER_NOISE = 6.0           # sigma = 6 x measured noise
DENOISE_SIGMA_RANGE = (25.0, 70.0)      # grid bounds of the sweep
CLAHE_PARAMS = dict(clip_limit=1.0, tile_grid=(8, 8), strength=0.5)
CLAHE_CONTRAST_MAX = 100                # CLAHE only if contrast after denoise is below this


def _gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def choose_denoise_sigma(img):
    """Adaptive rule: stronger denoising for noisier images (noise measured on the Retinex output)."""
    noise = estimate_noise(_gray(img))
    sigma = float(np.clip(DENOISE_SIGMA_PER_NOISE * noise, *DENOISE_SIGMA_RANGE))
    return sigma, noise


def _contrast(img):
    p5, p95 = np.percentile(_gray(img), [5, 95])
    return float(p95 - p5)


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

    adaptive_info = None
    with _timer(lat, "adaptive"):
        override = cfg.get("retinex_override")       # used by the sweep script only
        if override:
            rparams = {**RETINEX_PARAMS, **override}
        elif cfg["use_adaptive"]:
            rparams, adaptive_info = choose_retinex_params(stats, illum, RETINEX_PARAMS)
        else:
            rparams = dict(RETINEX_PARAMS)

    out = img
    with _timer(lat, "retinex"):
        if cfg["use_retinex"]:
            out = retinex_enhance(out, L=illum["illumination_map"], **rparams)
    retinex_out = out

    sigma, noise_meas = None, None
    with _timer(lat, "denoise"):
        if cfg["use_denoise"]:
            sigma, noise_meas = choose_denoise_sigma(out)
            denoised = denoise(out, "guided", sigma=sigma)
        else:
            denoised = out

    con = _contrast(denoised)
    clahe_applied = bool(cfg["use_clahe"] and con < CLAHE_CONTRAST_MAX)
    with _timer(lat, "clahe"):
        clahe_out = apply_clahe(denoised, **CLAHE_PARAMS) if clahe_applied else denoised

    cc_params, cc_measured = None, None
    with _timer(lat, "color_correction"):
        if cfg["use_color_correction"]:
            cc_params, cc_measured = choose_color_params(clahe_out)
            final = color_correct(clahe_out, **cc_params)
        else:
            final = clahe_out

    lat["total"] = round((time.perf_counter() - t_total) * 1000.0, 2)

    in_stats = _clean_stats(stats)
    in_stats["original_size"] = [int(orig_size[0]), int(orig_size[1])]
    in_stats["scale"] = round(float(scale), 4)

    meta = {
        "latency_ms": lat,
        "config": {
            "flags": cfg,
            "params": {
                "retinex": dict(rparams) if cfg["use_retinex"] else None,
                "denoise": ({"method": "guided", "sigma": round(sigma, 1),
                             "measured_noise": round(noise_meas, 2)} if sigma is not None else None),
                "clahe": {"applied": clahe_applied, "contrast_after_denoise": round(con, 1),
                          "contrast_max": CLAHE_CONTRAST_MAX,
                          **{k: (list(v) if isinstance(v, tuple) else v)
                             for k, v in CLAHE_PARAMS.items()}},
                "color_correction": ({**{k: round(v, 3) for k, v in cc_params.items()},
                                      **{k: round(v, 3) for k, v in cc_measured.items()}}
                                     if cc_params else None),
            },
            "adaptive": adaptive_info is not None,
            "adaptive_rule": adaptive_info,   # {feature, value, bin} or None
        },
        "stages_implemented": {"retinex": True, "clahe": True, "denoise": True,
                               "color_correction": True},
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