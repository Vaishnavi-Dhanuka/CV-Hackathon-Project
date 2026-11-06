FEATURES = {
    "mean_brightness": lambda s, i: float(s["mean_brightness"]),
    "illum_cv":        lambda s, i: float(i["illum_cv"]),
    "dark_fraction":   lambda s, i: float(s["dark_fraction"]),
}

# Chosen by tests/sweep_adaptive.py: fit on 97 images of LOL our485, verified on 97 different ones.
# Check-set result: 19.56 dB / 0.768 SSIM vs 19.25 dB for the fixed default (target_mean 0.5, gamma 0.5).
ADAPTIVE_RULE = {
    "feature": "mean_brightness",
    "edges": (11.2141, 16.0695),
    "params": [dict(target_mean=0.4, gamma=0.5),
               dict(target_mean=0.4, gamma=0.7),
               dict(target_mean=0.4, gamma=0.7)],
}


def choose_retinex_params(stats, illum, fixed):
    """Adaptive rule: measure one image property, pick its bin (0,1,2), use that bin's Retinex params.
    Returns (params, info). info is None when no rule is active."""
    if ADAPTIVE_RULE is None:
        return dict(fixed), None
    name = ADAPTIVE_RULE["feature"]
    v = FEATURES[name](stats, illum)
    e1, e2 = ADAPTIVE_RULE["edges"]
    b = 0 if v < e1 else (1 if v < e2 else 2)
    return {**fixed, **ADAPTIVE_RULE["params"][b]}, {"feature": name, "value": round(v, 4), "bin": b}