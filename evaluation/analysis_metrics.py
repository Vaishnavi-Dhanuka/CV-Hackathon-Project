"""Noise-suppression and structural-edge analysis metrics.

Supported images are NumPy arrays containing grayscale pixels (H, W) or
channels-last RGB pixels (H, W, 3). ``uint8`` values use [0, 255], while finite
floating-point values must be normalized to [0, 1]. Calculations use grayscale
intensities on a common [0, 255] scale.
"""

from __future__ import annotations

import cv2
import numpy as np


_MAD_TO_SIGMA = 1.4826
_SOBEL_EDGE_THRESHOLD = 0.1 * 4 * 255


def _validate_image(image: np.ndarray, name: str) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array.")
    if image.ndim not in (2, 3):
        raise ValueError(f"{name} must be a 2D grayscale or 3D RGB image.")
    if image.ndim == 3 and image.shape[-1] != 3:
        raise ValueError(f"{name} must have exactly three channels in RGB order.")
    if image.size == 0 or any(dimension == 0 for dimension in image.shape):
        raise ValueError(f"{name} must not be empty.")

    if image.dtype == np.uint8:
        return image.astype(np.float32)
    if np.issubdtype(image.dtype, np.floating):
        if not np.all(np.isfinite(image)):
            raise ValueError(f"{name} must contain only finite values.")
        if np.any(image < 0) or np.any(image > 1):
            raise ValueError(f"{name} float values must be normalized to [0, 1].")
        return image.astype(np.float32) * 255
    raise TypeError(f"{name} must use uint8 or float values normalized to [0, 1].")


def _grayscale_255(image: np.ndarray, name: str) -> np.ndarray:
    pixels = _validate_image(image, name)
    if pixels.ndim == 2:
        return pixels
    return cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)


def _validate_pair(
    reference: np.ndarray,
    enhanced: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    reference_gray = _grayscale_255(reference, "reference")
    enhanced_gray = _grayscale_255(enhanced, "enhanced")
    if reference_gray.shape != enhanced_gray.shape:
        raise ValueError(
            "reference and enhanced must have identical image dimensions; "
            f"got {reference_gray.shape} and {enhanced_gray.shape}."
        )
    return reference_gray, enhanced_gray


def _noise_level_from_grayscale(grayscale: np.ndarray) -> float:
    if min(grayscale.shape) < 3:
        raise ValueError("Noise estimation requires height and width of at least 3.")
    median_filtered = cv2.medianBlur(grayscale, 3)
    residual = grayscale - median_filtered
    residual_center = float(np.median(residual))
    mad = float(np.median(np.abs(residual - residual_center)))
    return _MAD_TO_SIGMA * mad


def estimate_noise_level(image: np.ndarray) -> float:
    """Estimate high-frequency residual scale with a robust MAD estimator.

    RGB images are converted to grayscale. A 3x3 median filter provides a
    local baseline; the median absolute deviation of the residual is multiplied
    by 1.4826, the normal-consistency factor for estimating standard deviation.
    The result is expressed in 8-bit grayscale intensity levels.

    This is a noise proxy, not a noise-only measurement: texture, fine detail,
    and edges can contribute to the residual. It assumes noise is represented
    primarily by local high-frequency deviations. A low estimate does not
    necessarily imply better restoration, since smoothing can remove useful
    image detail as well as noise.
    """
    return _noise_level_from_grayscale(_grayscale_255(image, "image"))


def calculate_noise_reduction(
    input_image: np.ndarray,
    enhanced_image: np.ndarray,
) -> float:
    """Return input noise proxy minus enhanced noise proxy.

    A positive result means the enhanced image has a lower MAD-based
    high-frequency residual estimate; a negative result means the estimate
    increased. The signed difference is in 8-bit grayscale intensity levels.
    It is not a standalone restoration-quality score: reducing the estimate
    through excessive smoothing may also remove useful detail, and texture or
    edges can affect either estimate.
    """
    input_gray, enhanced_gray = _validate_pair(input_image, enhanced_image)
    input_noise = _noise_level_from_grayscale(input_gray)
    enhanced_noise = _noise_level_from_grayscale(enhanced_gray)
    return input_noise - enhanced_noise


def calculate_edge_preservation(
    reference: np.ndarray,
    enhanced: np.ndarray,
) -> float:
    """Measure structural edge retention with a Sobel edge-mask F1 score.

    Grayscale Sobel magnitudes are thresholded at 10% of the maximum 3x3 Sobel
    response for an 8-bit intensity step. The returned score is the F1 overlap
    between reference and enhanced edge pixels (0 to 1); higher values mean
    more reference edges are retained while penalizing additional/missing
    edges. This exact-pixel overlap does not provide positional tolerance.

    If both images contain no detected edges, the score is defined as 1.0; if
    only one contains edges, it is 0.0. Inputs must have matching dimensions
    and may be grayscale or channels-last RGB. RGB is converted to grayscale.
    """
    reference_gray, enhanced_gray = _validate_pair(reference, enhanced)

    def edge_mask(grayscale: np.ndarray) -> np.ndarray:
        gradient_x = cv2.Sobel(grayscale, cv2.CV_32F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(grayscale, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(gradient_x, gradient_y)
        return magnitude >= _SOBEL_EDGE_THRESHOLD

    reference_edges = edge_mask(reference_gray)
    enhanced_edges = edge_mask(enhanced_gray)
    reference_count = int(np.count_nonzero(reference_edges))
    enhanced_count = int(np.count_nonzero(enhanced_edges))
    if reference_count == 0 and enhanced_count == 0:
        return 1.0
    if reference_count == 0 or enhanced_count == 0:
        return 0.0

    overlap = int(np.count_nonzero(reference_edges & enhanced_edges))
    precision = overlap / enhanced_count
    recall = overlap / reference_count
    return 2 * precision * recall / (precision + recall)
