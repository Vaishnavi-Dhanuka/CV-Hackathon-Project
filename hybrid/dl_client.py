"""Model-agnostic client for the LUMINA deep-learning enhancement API."""

import os
import time
from io import BytesIO
from typing import TypeAlias

import numpy as np
import requests
from PIL import Image, UnidentifiedImageError

API_BASE_URL: str = (
    os.getenv("LUMINA_DL_API_URL") or "http://127.0.0.1:8000"
).rstrip("/")

EnhancementResult: TypeAlias = tuple[np.ndarray, float]


def enhance_with_api(
    image: Image.Image, timeout: float = 120.0
) -> EnhancementResult:
    """Enhance an image through the configured HTTP API.

    The input is converted to RGB and uploaded as an in-memory PNG. The API
    response is decoded to an RGB ``uint8`` NumPy array, ready for fusion.

    Args:
        image: Input image to enhance.
        timeout: Maximum number of seconds to wait for the API response.

    Returns:
        A tuple containing the enhanced HxWx3 RGB array and API round-trip
        latency in seconds.

    Raises:
        TypeError: If ``image`` is not a Pillow image.
        ConnectionError: If the API cannot be reached.
        TimeoutError: If the API request times out.
        RuntimeError: If the API returns an unsuccessful status, an
            unexpected content type, or an invalid image response.
        ValueError: If the returned image dimensions differ from the input.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image.")
    if image.width == 0 or image.height == 0:
        raise ValueError("image must have non-empty dimensions.")

    input_size = image.size
    rgb_image = image.convert("RGB")
    image_bytes = BytesIO()
    rgb_image.save(image_bytes, format="PNG")

    endpoint = f"{API_BASE_URL}/enhance"
    started_at = time.perf_counter()
    try:
        response = requests.post(
            endpoint,
            files={
                "file": ("image.png", image_bytes.getvalue(), "image/png"),
            },
            timeout=timeout,
        )
    except requests.exceptions.Timeout as error:
        raise TimeoutError(
            f"Timed out while requesting image enhancement from {endpoint} "
            f"after {timeout} seconds."
        ) from error
    except requests.exceptions.ConnectionError as error:
        raise ConnectionError(
            f"Could not connect to the image enhancement API at {endpoint}."
        ) from error
    except requests.exceptions.RequestException as error:
        raise RuntimeError(
            f"Image enhancement API request to {endpoint} failed: {error}."
        ) from error

    latency_seconds = time.perf_counter() - started_at

    if not response.ok:
        raise RuntimeError(
            "Image enhancement API returned unsuccessful HTTP response "
            f"{response.status_code} {response.reason}."
        )

    content_type = response.headers.get("Content-Type", "")
    media_type = content_type.split(";", maxsplit=1)[0].strip().lower()
    if media_type != "image/png":
        raise RuntimeError(
            "Image enhancement API returned unexpected content type "
            f"{content_type or '<missing>'}; expected image/png."
        )

    try:
        with Image.open(BytesIO(response.content)) as enhanced_image:
            if enhanced_image.format != "PNG":
                raise RuntimeError(
                    "Image enhancement API declared image/png but returned "
                    f"{enhanced_image.format or 'an unknown image format'}."
                )
            enhanced_rgb = enhanced_image.convert("RGB")
            enhanced_array = np.array(enhanced_rgb, dtype=np.uint8, copy=True)
    except (
        UnidentifiedImageError,
        Image.DecompressionBombError,
        OSError,
        ValueError,
    ) as error:
        raise RuntimeError(
            "Image enhancement API returned an invalid PNG image response."
        ) from error

    if enhanced_array.shape[:2] != (input_size[1], input_size[0]):
        returned_size = (enhanced_array.shape[1], enhanced_array.shape[0])
        raise ValueError(
            "Image enhancement API returned dimensions "
            f"{returned_size}, but the input dimensions were {input_size}."
        )

    return enhanced_array, latency_seconds
