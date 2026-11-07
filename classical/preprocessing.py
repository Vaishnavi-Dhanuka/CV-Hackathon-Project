import os
import cv2
import numpy as np

MIN_SIDE = 16      # smaller than this: reject
MAX_SIDE = 1024    # larger than this: downscale (keeps runtime < ~10 s)


def to_rgb_uint8(image):
    """Validate any array and return uint8 RGB (H, W, 3). Raises ValueError if unusable."""
    if image is None:
        raise ValueError("Image is None (file missing or unreadable)")
    if not isinstance(image, np.ndarray) or image.size == 0 or image.ndim not in (2, 3):
        raise ValueError("Image must be a non-empty 2D or 3D numpy array")

    # --- fix dtype ---
    if image.dtype != np.uint8:
        if image.dtype == np.uint16:
            image = image / 257.0
        elif np.issubdtype(image.dtype, np.floating):
            image = image * 255.0 if image.max() <= 1.0 else image
        image = np.rint(np.clip(image, 0, 255)).astype(np.uint8)

    # --- fix channels ---
    if image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0]
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = image[:, :, :3]          # drop alpha
    elif image.shape[2] != 3:
        raise ValueError(f"Unsupported channel count: {image.shape[2]}")

    h, w = image.shape[:2]
    if min(h, w) < MIN_SIDE:
        raise ValueError(f"Image too small ({w}x{h}); minimum side is {MIN_SIDE}px")
    return np.ascontiguousarray(image)


def load_image(path):
    """Read a file from disk -> uint8 RGB. The ONE BGR->RGB conversion happens here."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)   # keeps alpha / 16-bit
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    return to_rgb_uint8(img)


def save_image(image, path):
    """Write uint8 RGB to disk. The ONE RGB->BGR conversion happens here."""
    folder = os.path.dirname(str(path))
    if folder:
        os.makedirs(folder, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def resize_if_large(image, max_side=MAX_SIDE):
    """Downscale so the longest side <= max_side. Returns (image, scale)."""
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return image, 1.0
    scale = max_side / longest
    new_size = (round(w * scale), round(h * scale))   # (width, height)
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA), scale


def estimate_noise(gray):
    """Immerkaer noise sigma estimate on a uint8 gray image."""
    g = gray.astype(np.float32)
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    h, w = g.shape
    conv = cv2.filter2D(g, -1, kernel)[1:-1, 1:-1]
    return float(np.sqrt(np.pi / 2) * np.abs(conv).sum() / (6.0 * (h - 2) * (w - 2)))


def compute_characteristics(image):
    """Measure the image so the pipeline can pick parameters."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    g = gray.astype(np.float32)

    p5, p95 = np.percentile(g, [5, 95])
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    hist = hist / hist.sum()

    pixels = image.reshape(-1, 3).astype(np.float32)
    ch_mean = pixels.mean(axis=0)            # order: R, G, B
    ch_std = pixels.std(axis=0)

    return {
        "mean_brightness": float(g.mean()),
        "std": float(g.std()),
        "contrast": float(p95 - p5),
        "histogram": hist,                    # 256 values summing to 1
        "channel_mean": ch_mean,              # R, G, B
        "channel_std": ch_std,
        "color_cast": float((ch_mean.max() - ch_mean.min()) / (ch_mean.mean() + 1e-6)),
        "dark_fraction": float((gray < 50).mean()),
        "highlight_fraction": float((gray > 240).mean()),
        "noise_sigma": estimate_noise(gray),
    }


def prepare_image(source, max_side=MAX_SIDE):
    """Main entry point. source = file path OR numpy array.
    Returns (image_uint8_rgb, info_dict)."""
    if isinstance(source, (str, os.PathLike)):
        img = load_image(source)
    else:
        img = to_rgb_uint8(source)
    original_size = img.shape[:2]
    img, scale = resize_if_large(img, max_side)
    stats = compute_characteristics(img)
    stats["original_size"] = original_size
    stats["scale"] = scale
    return img, stats