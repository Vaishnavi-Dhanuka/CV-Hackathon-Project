"""Adapter for the M1 classical enhancement package.

The ``classical`` package must be installed or available on the Python import
path in the project environment.
"""

import time

import cv2
import numpy as np
from PIL import Image

from classical.preprocessing import prepare_image
from classical.retinex import retinex_enhance


def enhance_with_classical(image: Image.Image) -> tuple[np.ndarray, float]:
    """Run M1 preprocessing and guided Retinex enhancement on a PIL image.

    Args:
        image: A non-empty Pillow image in any supported mode.

    Returns:
        A tuple of the enhanced contiguous HxWx3 RGB uint8 array at the
        original dimensions, and processing latency in seconds.

    Raises:
        TypeError: If the input is not a Pillow image or M1 returns an invalid
            type or dtype.
        ValueError: If the input is empty or the M1 output has invalid
            dimensions.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image.")
    if image.width == 0 or image.height == 0:
        raise ValueError("image must have non-empty dimensions.")

    original_size = (image.width, image.height)
    started_at = time.perf_counter()
    rgb_image = image.convert("RGB")
    rgb_array = np.array(rgb_image, dtype=np.uint8, copy=True)

    prepared_image, _ = prepare_image(rgb_array)
    enhanced_image = retinex_enhance(prepared_image)

    if not isinstance(enhanced_image, np.ndarray):
        raise TypeError("retinex_enhance must return a NumPy array.")
    if enhanced_image.dtype != np.uint8:
        raise TypeError("retinex_enhance must return an array with dtype uint8.")
    if enhanced_image.ndim != 3 or enhanced_image.shape[2] != 3:
        raise ValueError("retinex_enhance must return an HxWx3 RGB image.")
    if enhanced_image.shape[:2] != prepared_image.shape[:2]:
        raise ValueError(
            "retinex_enhance output dimensions must match the preprocessed image."
        )

    if prepared_image.shape[:2] != (original_size[1], original_size[0]):
        enhanced_image = cv2.resize(
            enhanced_image,
            original_size,
            interpolation=cv2.INTER_LINEAR,
        )

    enhanced_array = np.ascontiguousarray(enhanced_image, dtype=np.uint8)
    latency_seconds = time.perf_counter() - started_at
    return enhanced_array, latency_seconds
