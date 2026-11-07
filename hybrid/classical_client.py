"""Adapter for the M1 classical enhancement package.

The ``classical`` package must be installed or available on the Python import
path in the project environment.
"""

import time

import cv2
import numpy as np
from PIL import Image

from classical.classical_pipeline import enhance_classical


def enhance_with_classical(image: Image.Image) -> tuple[np.ndarray, float]:
    """Run the full M1 pipeline and return its final image and latency."""
    enhanced_image, latency_seconds, _, _ = enhance_with_classical_artifacts(image)
    return enhanced_image, latency_seconds


def enhance_with_classical_artifacts(
    image: Image.Image,
) -> tuple[np.ndarray, float, dict[str, np.ndarray], dict[str, object]]:
    """Run M1 and return the final image, latency, display artifacts, and metadata.

    Args:
        image: A non-empty Pillow image in any supported mode.

    Returns:
        Final contiguous HxWx3 RGB uint8 image at the original dimensions,
        processing latency in seconds, available uint8 RGB intermediate images,
        and pipeline metadata.

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
    rgb_array = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
    enhanced_image, intermediate = enhance_classical(rgb_array)
    if not isinstance(enhanced_image, np.ndarray):
        raise TypeError("enhance_classical must return a NumPy final image.")
    if enhanced_image.dtype != np.uint8:
        raise TypeError("enhance_classical final image must have dtype uint8.")
    if enhanced_image.ndim != 3 or enhanced_image.shape[2] != 3:
        raise ValueError("enhance_classical final image must be HxWx3 RGB.")
    if not isinstance(intermediate, dict):
        raise TypeError("enhance_classical must return an intermediate dictionary.")

    required_artifacts = (
        "illumination",
        "reflectance",
        "retinex",
        "clahe",
        "denoised",
        "final",
    )
    artifacts: dict[str, np.ndarray] = {}
    for name in required_artifacts:
        artifact = intermediate.get(name)
        if not isinstance(artifact, np.ndarray):
            raise TypeError(
                f"enhance_classical intermediate {name!r} must be a NumPy array."
            )
        if artifact.dtype != np.uint8:
            raise TypeError(
                f"enhance_classical intermediate {name!r} must have dtype uint8."
            )
        if artifact.ndim != 3 or artifact.shape[2] != 3:
            raise ValueError(
                f"enhance_classical intermediate {name!r} must be HxWx3 RGB."
            )
        if artifact.shape != enhanced_image.shape:
            raise ValueError(
                f"enhance_classical intermediate {name!r} must match the final "
                "image dimensions."
            )
        artifacts[name] = artifact
    if not np.array_equal(artifacts["final"], enhanced_image):
        raise ValueError(
            "enhance_classical returned different final and intermediate images."
        )

    if enhanced_image.shape[:2] != (original_size[1], original_size[0]):
        enhanced_image = cv2.resize(
            enhanced_image,
            original_size,
            interpolation=cv2.INTER_LINEAR,
        )
        artifacts = {
            name: cv2.resize(
                artifact,
                original_size,
                interpolation=cv2.INTER_LINEAR,
            )
            for name, artifact in artifacts.items()
        }

    enhanced_array = np.ascontiguousarray(enhanced_image, dtype=np.uint8)
    contiguous_artifacts = {
        name: np.ascontiguousarray(artifact, dtype=np.uint8)
        for name, artifact in artifacts.items()
    }
    latency_seconds = time.perf_counter() - started_at
    metadata = intermediate.get("meta")
    if not isinstance(metadata, dict):
        raise TypeError("enhance_classical intermediate metadata must be a dictionary.")
    return enhanced_array, latency_seconds, contiguous_artifacts, metadata
