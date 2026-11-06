import cv2
import numpy as np
from classical.illumination import estimate_illumination


def _gray_mean(x):
    """Mean brightness of a float RGB image in [0, 1]."""
    return float((0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]).mean())


def _gaussian(img, sigma):
    """Gaussian blur; big sigmas are done on a shrunken copy (same result, much faster)."""
    if sigma <= 30:
        return cv2.GaussianBlur(img, (0, 0), sigma)
    s = 20.0 / sigma
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 20)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def _stretch(x, low=1, high=99):
    """Map log-domain output to [0, 1] with ONE shared range (keeps colour ratios)."""
    lo, hi = np.percentile(x, [low, high])
    return np.clip((x - lo) / (hi - lo + 1e-6), 0, 1)


def single_scale_retinex(image, sigma=80):
    """SSR: log(I) - log(Gaussian(I)). Returns stretched float RGB in [0, 1]."""
    img = image.astype(np.float32) + 1.0               # +1 avoids log(0)
    out = np.log(img) - np.log(_gaussian(img, sigma))
    return _stretch(out)


def multi_scale_retinex(image, sigmas=(15, 80, 250)):
    """MSR: average SSR over small / medium / large scales."""
    img = image.astype(np.float32) + 1.0
    logi = np.log(img)
    out = sum(logi - np.log(_gaussian(img, s)) for s in sigmas) / len(sigmas)
    return _stretch(out)


def illumination_correction(image, L=None, gamma=0.5, max_gain=8.0):
    """Output = R * L^gamma = I * L^(gamma-1). Edge-aware because L comes from the guided filter."""
    img = image.astype(np.float32) / 255.0
    if L is None:
        L = estimate_illumination(image)
    gain = np.clip(L ** (gamma - 1.0), 1.0, max_gain)   # always >= 1 (never darkens)
    return img * gain[:, :, None], gain


def restore_brightness(x, target_mean=0.42, max_gain=4.0):
    """Scale up toward a target mean brightness. Never darkens an already bright image."""
    m = _gray_mean(np.clip(x, 0, 1))
    g = float(np.clip(target_mean / (m + 1e-6), 1.0, max_gain))
    return x * g


def soft_clip(x, knee=0.8):
    """Smooth highlight roll-off above `knee` instead of a hard clip at 1."""
    x = np.maximum(x, 0.0)
    out = x.copy()
    over = x > knee
    out[over] = knee + (1 - knee) * np.tanh((x[over] - knee) / (1 - knee))
    return out


def retinex_enhance(image, method="guided", gamma=0.5, max_gain=8.0,
                    target_mean=0.42, strength=0.9, L=None):
    """Main entry point. image: uint8 RGB. method: 'guided' (ours), 'msr' or 'ssr'.
    Returns uint8 RGB."""
    img = image.astype(np.float32) / 255.0
    if method == "ssr":
        out = single_scale_retinex(image)
    elif method == "msr":
        out = multi_scale_retinex(image)
    else:
        out, _ = illumination_correction(image, L, gamma, max_gain)
        out = restore_brightness(out, target_mean)
    out = soft_clip(out)
    out = strength * out + (1.0 - strength) * img          # blend with the original
    return (np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8)