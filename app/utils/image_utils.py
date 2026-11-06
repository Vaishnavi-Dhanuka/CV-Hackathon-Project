"""Validated image loading and preliminary, non-model image analysis.

Brightness labels use fixed grayscale-mean thresholds: very dark < 64, dark
< 128, moderate < 192, and well illuminated >= 192 (on a 0-255 scale). These
are convenience heuristics, not ground-truth scene classifications.
"""

from io import BytesIO
from typing import TypedDict

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
BRIGHTNESS_THRESHOLDS = (64.0, 128.0, 192.0)


class ImageAnalysis(TypedDict):
    """Measurements derived directly from a validated RGB image."""

    width: int
    height: int
    dimensions: tuple[int, int]
    brightness: float
    brightness_classification: str
    contrast: float
    sharpness: float
    noise_proxy: float
    channel_means: dict[str, float]


def _classify_brightness(brightness: float) -> str:
    if brightness < BRIGHTNESS_THRESHOLDS[0]:
        return "Very dark"
    if brightness < BRIGHTNESS_THRESHOLDS[1]:
        return "Dark"
    if brightness < BRIGHTNESS_THRESHOLDS[2]:
        return "Moderate"
    return "Well illuminated"


def load_and_analyze_image(
    image_bytes: bytes,
    filename: str,
) -> tuple[Image.Image, ImageAnalysis]:
    """Validate an upload, return RGB pixels, and calculate image measurements."""
    extension = filename[filename.rfind(".") :].lower() if "." in filename else ""
    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError("Only JPG, JPEG, and PNG files are supported.")
    if not image_bytes:
        raise ValueError("The uploaded file is empty.")

    try:
        with Image.open(BytesIO(image_bytes)) as source:
            if source.format not in {"JPEG", "PNG"}:
                raise ValueError("The uploaded file content must be a JPEG or PNG image.")
            source.verify()
        with Image.open(BytesIO(image_bytes)) as source:
            image = source.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ValueError("The file is not a valid, readable image.") from error

    rgb = np.asarray(image)
    grayscale = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray_float = grayscale.astype(np.float32)
    median_filtered = cv2.medianBlur(grayscale, 3).astype(np.float32)
    high_frequency_residual = gray_float - median_filtered
    residual_center = float(np.median(high_frequency_residual))
    noise_proxy = 1.4826 * float(
        np.median(np.abs(high_frequency_residual - residual_center))
    )
    brightness = float(np.mean(grayscale))
    height, width = grayscale.shape

    analysis: ImageAnalysis = {
        "width": width,
        "height": height,
        "dimensions": (width, height),
        "brightness": brightness,
        "brightness_classification": _classify_brightness(brightness),
        "contrast": float(np.std(grayscale)),
        "sharpness": float(cv2.Laplacian(grayscale, cv2.CV_64F).var()),
        "noise_proxy": noise_proxy,
        "channel_means": {
            "red": float(np.mean(rgb[:, :, 0])),
            "green": float(np.mean(rgb[:, :, 1])),
            "blue": float(np.mean(rgb[:, :, 2])),
        },
    }
    return image, analysis
