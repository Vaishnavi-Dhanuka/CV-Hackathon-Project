"""Baseline pixel-level fusion for classical and deep-learning restorations."""

import cv2
import numpy as np


def _validate_rgb_uint8_image(image: np.ndarray, name: str) -> None:
    if not isinstance(image, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array.")
    if image.dtype != np.uint8:
        raise TypeError(f"{name} must have dtype uint8.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"{name} must be an HxWx3 RGB image.")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError(f"{name} must have non-empty image dimensions.")


def weighted_fusion(
    classical_image: np.ndarray,
    dl_image: np.ndarray,
    classical_weight: float = 0.5,
) -> np.ndarray:
    """Blend two same-size uint8 RGB restoration results by weighted averaging.

    ``classical_weight`` controls the classical contribution; the deep-learning
    contribution is ``1 - classical_weight``. The result is rounded to the
    nearest integer, clipped to the uint8 range, and returned as an RGB array.

    This is a baseline, fixed-weight fusion method, not the project's final
    adaptive fusion algorithm. It combines already-restored outputs and does
    not itself imply or guarantee a quality improvement.

    Args:
        classical_image: HxWx3 uint8 RGB classical restoration output.
        dl_image: HxWx3 uint8 RGB deep-learning restoration output.
        classical_weight: Finite weight in [0, 1] for the classical output.

    Returns:
        The weighted fusion as an HxWx3 uint8 RGB NumPy array.

    Raises:
        TypeError: If an image is not a NumPy array, an image is not uint8, or
            the weight is not a real numeric value.
        ValueError: If the images have invalid/mismatched dimensions or the
            weight is non-finite or outside [0, 1].
    """
    _validate_rgb_uint8_image(classical_image, "classical_image")
    _validate_rgb_uint8_image(dl_image, "dl_image")

    if classical_image.shape != dl_image.shape:
        raise ValueError(
            "classical_image and dl_image must have identical dimensions."
        )
    if isinstance(classical_weight, (bool, np.bool_)) or not isinstance(
        classical_weight, (int, float, np.integer, np.floating)
    ):
        raise TypeError("classical_weight must be a real numeric value.")

    weight = float(classical_weight)
    if not np.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError("classical_weight must be finite and between 0 and 1.")

    blended = (
        weight * classical_image.astype(np.float32)
        + (1.0 - weight) * dl_image.astype(np.float32)
    )
    return np.clip(np.rint(blended), 0, 255).astype(np.uint8)


def adaptive_fusion(
    classical_image: np.ndarray,
    dl_image: np.ndarray,
    input_image: np.ndarray,
    return_weights: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, float | dict[str, float]]]:
    """Fuse restoration outputs using a deterministic input-statistics prior.

    All inputs must be same-size HxWx3 RGB NumPy arrays with ``uint8`` dtype.
    Input luminance uses Rec. 709 coefficients and the [0, 255] scale.

    The classical weight starts at 0.5 and is adjusted by centered, clipped
    indicators:

    * darkness = clip((128 - mean luminance) / 128, 0, 1)
    * underexposure = fraction of pixels with luminance below 64
    * low contrast = clip((64 - luminance standard deviation) / 64, 0, 1)
    * classical weight = clip(0.5 + 0.15*(darkness - 0.5)
      + 0.15*(underexposure - 0.5) + 0.10*(low contrast - 0.5), 0.25, 0.75)

    The DL weight is the remainder, so weights are non-negative and sum to
    one. This bounded, global rule is a preliminary engineering heuristic: its
    direction is not evidence that either restoration method is superior, and
    image statistics alone do not measure output quality. It is not a learned
    or locally adaptive fusion method.

    Args:
        classical_image: HxWx3 uint8 RGB classical restoration output.
        dl_image: HxWx3 uint8 RGB deep-learning restoration output.
        input_image: Original low-light HxWx3 uint8 RGB input.
        return_weights: If true, also return weights and input statistics.

    Returns:
        The fused HxWx3 uint8 RGB image, or a pair containing that image and a
        dictionary with ``classical_weight``, ``dl_weight``, and nested
        ``input_statistics`` (mean luminance, contrast, and underexposed
        fraction).

    Raises:
        TypeError: If an input is not a NumPy array or is not uint8, or
            ``return_weights`` is not a boolean.
        ValueError: If an input is not a non-empty RGB image or dimensions
            differ.
    """
    _validate_rgb_uint8_image(classical_image, "classical_image")
    _validate_rgb_uint8_image(dl_image, "dl_image")
    _validate_rgb_uint8_image(input_image, "input_image")

    if classical_image.shape != dl_image.shape:
        raise ValueError(
            "classical_image and dl_image must have identical dimensions."
        )
    if input_image.shape != classical_image.shape:
        raise ValueError(
            "classical_image, dl_image, and input_image must have identical "
            "dimensions."
        )
    if not isinstance(return_weights, (bool, np.bool_)):
        raise TypeError("return_weights must be a boolean.")

    rgb = input_image.astype(np.float32)
    luminance = (
        0.2126 * rgb[:, :, 0]
        + 0.7152 * rgb[:, :, 1]
        + 0.0722 * rgb[:, :, 2]
    )
    mean_luminance = float(np.mean(luminance))
    luminance_contrast = float(np.std(luminance))
    underexposed_fraction = float(np.mean(luminance < 64.0))

    darkness = float(np.clip((128.0 - mean_luminance) / 128.0, 0.0, 1.0))
    low_contrast = float(np.clip((64.0 - luminance_contrast) / 64.0, 0.0, 1.0))
    classical_weight = float(
        np.clip(
            0.5
            + 0.15 * (darkness - 0.5)
            + 0.15 * (underexposed_fraction - 0.5)
            + 0.10 * (low_contrast - 0.5),
            0.25,
            0.75,
        )
    )
    dl_weight = 1.0 - classical_weight
    fused = weighted_fusion(classical_image, dl_image, classical_weight)

    if not return_weights:
        return fused

    return fused, {
        "classical_weight": classical_weight,
        "dl_weight": dl_weight,
        "input_statistics": {
            "mean_luminance": mean_luminance,
            "luminance_contrast": luminance_contrast,
            "underexposed_fraction": underexposed_fraction,
        },
    }


def spatial_adaptive_fusion(
    classical_image: np.ndarray,
    dl_image: np.ndarray,
    input_image: np.ndarray,
    return_weights: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    """Blend restoration outputs with smoothly varying, bounded local weights.

    The global baseline and input statistics come from :func:`adaptive_fusion`.
    RGB input luminance is computed with Rec. 709 coefficients and normalized
    to [0, 1]. A Gaussian blur creates a smooth local illumination estimate.
    Before the final weight smoothing, each location adjusts the baseline
    classical weight by ``0.20 * (0.5 - local_illumination)``. Thus even the
    darkest/brightest estimates can change the baseline by at most 0.10;
    darker regions get a modest classical-weight increase and brighter regions
    a decrease. The weight map is clipped to [0.25, 0.75], then Gaussian
    smoothed and clipped again. The DL map is exactly ``1 - classical_map``.

    The illumination blur uses sigma ``clip(min(height, width) / 32, 0.5, 8)``
    and the weight-map blur uses sigma ``clip(min(height, width) / 16, 1, 8)``
    pixels, with replicated borders. These bounded scales keep tiny images
    valid while smoothing local transitions. This is a deterministic
    heuristic, not a learned fusion method. It is not quality-calibrated, and
    local input brightness alone cannot establish which enhancement output is
    preferable.

    Args:
        classical_image: HxWx3 uint8 RGB classical restoration output.
        dl_image: HxWx3 uint8 RGB deep-learning restoration output.
        input_image: Original low-light HxWx3 uint8 RGB input.
        return_weights: If true, also return local maps and global baseline
            statistics and weights.

    Returns:
        The fused HxWx3 uint8 RGB image, or a pair containing the image and a
        dictionary with ``classical_weight_map``, ``dl_weight_map``,
        ``input_statistics``, and ``global_baseline_weights``.

    Raises:
        TypeError: If an input is not a NumPy array or is not uint8, or
            ``return_weights`` is not a boolean.
        ValueError: If an input is not a non-empty RGB image or dimensions
            differ.
    """
    _validate_rgb_uint8_image(classical_image, "classical_image")
    _validate_rgb_uint8_image(dl_image, "dl_image")
    _validate_rgb_uint8_image(input_image, "input_image")

    if classical_image.shape != dl_image.shape:
        raise ValueError(
            "classical_image and dl_image must have identical dimensions."
        )
    if input_image.shape != classical_image.shape:
        raise ValueError(
            "classical_image, dl_image, and input_image must have identical "
            "dimensions."
        )
    if not isinstance(return_weights, (bool, np.bool_)):
        raise TypeError("return_weights must be a boolean.")

    _, baseline = adaptive_fusion(
        classical_image, dl_image, input_image, return_weights=True
    )
    global_classical_weight = baseline["classical_weight"]
    height, width = input_image.shape[:2]
    shortest_side = min(height, width)

    rgb = input_image.astype(np.float32)
    luminance = (
        0.2126 * rgb[:, :, 0]
        + 0.7152 * rgb[:, :, 1]
        + 0.0722 * rgb[:, :, 2]
    ) / 255.0
    illumination_sigma = max(0.5, min(8.0, shortest_side / 32.0))
    local_illumination = cv2.GaussianBlur(
        luminance,
        (0, 0),
        sigmaX=illumination_sigma,
        borderType=cv2.BORDER_REPLICATE,
    )

    classical_weight_map = global_classical_weight + 0.20 * (
        0.5 - local_illumination
    )
    weight_sigma = max(1.0, min(8.0, shortest_side / 16.0))
    classical_weight_map = cv2.GaussianBlur(
        classical_weight_map.astype(np.float32),
        (0, 0),
        sigmaX=weight_sigma,
        borderType=cv2.BORDER_REPLICATE,
    )
    classical_weight_map = np.clip(classical_weight_map, 0.25, 0.75)
    dl_weight_map = 1.0 - classical_weight_map

    blended = (
        classical_weight_map[:, :, None] * classical_image.astype(np.float32)
        + dl_weight_map[:, :, None] * dl_image.astype(np.float32)
    )
    fused = np.clip(np.rint(blended), 0, 255).astype(np.uint8)
    if not return_weights:
        return fused

    return fused, {
        "classical_weight_map": classical_weight_map,
        "dl_weight_map": dl_weight_map,
        "input_statistics": baseline["input_statistics"],
        "global_baseline_weights": {
            "classical_weight": baseline["classical_weight"],
            "dl_weight": baseline["dl_weight"],
        },
    }
