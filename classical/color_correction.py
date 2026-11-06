import numpy as np

# Starting values. They are NOT tuned yet: the sweep in the next step picks the final ones on our485.
CAST_LOW, CAST_HIGH = 0.05, 0.20     # cast below LOW: leave colours alone; above HIGH: full correction
GAIN_RANGE = (0.75, 1.33)            # per-channel white-balance gain limits (stops wild corrections)
GAMMA_RANGE = (0.8, 1.1)             # adaptive gamma limits (mostly brightens, barely darkens)
TARGET_MEAN = 0.5                    # same exposure target as Retinex
CONTRAST_RANGE_MIN = 0.7             # p99-p1 spread (0..1) below this counts as "flat"
KNEE_NORMAL, KNEE_BRIGHT = 0.85, 0.75


def color_cast(image):
    """Cast score = (max channel mean - min channel mean) / overall mean. 0 = neutral."""
    m = image.reshape(-1, 3).astype(np.float32).mean(axis=0)
    return float((m.max() - m.min()) / (m.mean() + 1e-6))


def gray_world(x, strength=1.0):
    """White balance. x: float RGB in [0,1]. Assumes the scene averages to gray, so each channel
    is scaled toward the common mean. Near-black and near-white pixels are ignored (they carry
    no colour information). Gains are clipped and blended by `strength`."""
    if strength <= 0:
        return x
    gray = x.mean(axis=2)
    mask = (gray > 0.04) & (gray < 0.96)
    if mask.sum() < 100:
        return x
    means = x[mask].mean(axis=0)
    gains = np.clip(means.mean() / (means + 1e-6), *GAIN_RANGE)
    gains = 1.0 + strength * (gains - 1.0)
    return x * gains


def adaptive_gamma_value(mean_brightness, target=TARGET_MEAN):
    """Gamma that would move the mean to the target: mean**g = target -> g = ln(target)/ln(mean)."""
    m = float(np.clip(mean_brightness, 1e-3, 0.999))
    return float(np.clip(np.log(target) / np.log(m), *GAMMA_RANGE))


def apply_gamma(x, gamma):
    return np.power(np.clip(x, 0, 1), gamma) if abs(gamma - 1.0) > 1e-3 else x


def contrast_stretch(x, strength=0.5, low=1, high=99):
    """Stretch the p1..p99 range of brightness to 0..1 with ONE shared range (keeps colours)."""
    if strength <= 0:
        return x
    lo, hi = np.percentile(x.mean(axis=2), [low, high])
    stretched = (x - lo) / (hi - lo + 1e-6)
    return strength * stretched + (1.0 - strength) * x


def highlight_protect(x, knee=KNEE_NORMAL):
    """Roll off the brightest pixels smoothly. Works on the max channel and rescales the whole
    pixel, so hue is preserved (per-channel clipping would turn bright areas pink/yellow)."""
    m = x.max(axis=2)
    over = m > knee
    if not over.any():
        return x
    new_m = knee + (1 - knee) * np.tanh((m - knee) / (1 - knee))
    scale = np.where(over, new_m / (m + 1e-6), 1.0)
    return x * scale[:, :, None]


def choose_color_params(image):
    """Adaptive rule. Measures the image and returns (params, measured).
    params go straight into color_correct(); measured goes into meta for the UI."""
    x = image.astype(np.float32) / 255.0
    gray = x.mean(axis=2)
    cast = color_cast(image)
    mean_b = float(gray.mean())
    p1, p99 = np.percentile(gray, [1, 99])
    spread = float(p99 - p1)
    hi_frac = float((gray > 0.94).mean())

    params = {
        "wb_strength": float(np.clip((cast - CAST_LOW) / (CAST_HIGH - CAST_LOW), 0, 1)),
        "gamma": adaptive_gamma_value(mean_b),
        "contrast_strength": 0.5 if spread < CONTRAST_RANGE_MIN else 0.0,
        "knee": KNEE_BRIGHT if hi_frac > 0.02 else KNEE_NORMAL,
    }
    measured = {"color_cast": cast, "mean_brightness": mean_b,
                "brightness_spread": spread, "highlight_fraction": hi_frac}
    return params, measured


def color_correct(image, wb_strength=1.0, gamma=1.0, contrast_strength=0.5, knee=KNEE_NORMAL):
    """Main entry point. uint8 RGB in, uint8 RGB out.
    Order: white balance -> gamma -> contrast -> highlight protection."""
    x = image.astype(np.float32) / 255.0
    x = gray_world(x, wb_strength)
    x = apply_gamma(x, gamma)
    x = contrast_stretch(x, contrast_strength)
    x = highlight_protect(np.maximum(x, 0.0), knee)
    return (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)
