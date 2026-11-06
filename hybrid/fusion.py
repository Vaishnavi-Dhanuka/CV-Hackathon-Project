"""Baseline pixel-level fusion for classical and deep-learning restorations."""

import numpy as np


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
    images = (
        ("classical_image", classical_image),
        ("dl_image", dl_image),
    )
    for name, image in images:
        if not isinstance(image, np.ndarray):
            raise TypeError(f"{name} must be a NumPy array.")
        if image.dtype != np.uint8:
            raise TypeError(f"{name} must have dtype uint8.")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"{name} must be an HxWx3 RGB image.")
        if image.shape[0] == 0 or image.shape[1] == 0:
            raise ValueError(f"{name} must have non-empty image dimensions.")

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
