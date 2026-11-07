"""Reference-based image quality metrics for paired image evaluation.

Inputs must be NumPy arrays containing either two-dimensional grayscale images
or channels-last RGB images. ``uint8`` values use a data range of 255. Floating
point inputs must be finite and normalized to [0, 1], which uses a data range
of 1.0. Other dtypes and value ranges are rejected rather than guessed.
"""

from __future__ import annotations

import math
from threading import Lock
from typing import Any

import numpy as np
from skimage.metrics import structural_similarity

_LPIPS_MODEL: tuple[Any, Any, Any] | None = None
_LPIPS_MODEL_LOCK = Lock()
_NIQE_METRIC: tuple[Any, Any] | None = None
_NIQE_METRIC_LOCK = Lock()


def _validate_pair(
    reference: np.ndarray,
    enhanced: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, int | None]:
    """Validate a grayscale or RGB pair and return its metric data range."""
    if not isinstance(reference, np.ndarray) or not isinstance(enhanced, np.ndarray):
        raise TypeError("reference and enhanced must both be NumPy arrays.")
    if reference.ndim not in (2, 3) or enhanced.ndim not in (2, 3):
        raise ValueError("Images must be 2D grayscale or 3D channels-last RGB arrays.")
    if reference.shape != enhanced.shape:
        raise ValueError(
            "reference and enhanced must have identical dimensions; "
            f"got {reference.shape} and {enhanced.shape}."
        )
    if reference.ndim == 3 and reference.shape[-1] != 3:
        raise ValueError(
            "3D images must have exactly three channels in channels-last RGB order."
        )
    if reference.size == 0 or any(dimension == 0 for dimension in reference.shape):
        raise ValueError("Images must not be empty.")

    reference_is_uint8 = reference.dtype == np.uint8
    enhanced_is_uint8 = enhanced.dtype == np.uint8
    reference_is_float = np.issubdtype(reference.dtype, np.floating)
    enhanced_is_float = np.issubdtype(enhanced.dtype, np.floating)
    if reference_is_uint8 and enhanced_is_uint8:
        data_range = 255.0
    elif reference_is_float and enhanced_is_float:
        data_range = 1.0
        if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(enhanced)):
            raise ValueError("Floating-point images must contain only finite values.")
        if (
            np.any(reference < 0)
            or np.any(reference > 1)
            or np.any(enhanced < 0)
            or np.any(enhanced > 1)
        ):
            raise ValueError(
                "Floating-point image values must be normalized to the [0, 1] range."
            )
    else:
        raise TypeError(
            "Both images must use uint8 or both must use floating-point values "
            "normalized to [0, 1]."
        )

    channel_axis = -1 if reference.ndim == 3 else None
    return reference, enhanced, data_range, channel_axis


def calculate_psnr(reference: np.ndarray, enhanced: np.ndarray) -> float:
    """Calculate peak signal-to-noise ratio (PSNR) for a reference image pair.

    PSNR expresses the mean squared pixel error on a logarithmic decibel scale;
    a larger value indicates less pixel-wise error. A reference image is
    required because PSNR compares each enhanced pixel with its corresponding
    ground-truth pixel and is not an opinion score for a single image.

    Accepts matching 2D grayscale or channels-last RGB NumPy arrays. ``uint8``
    images use data range 255. Floating-point images must be finite and
    normalized to [0, 1], using data range 1.0. Identical images return
    ``math.inf`` because their mean squared error is zero.

    Raises:
        TypeError: If inputs are not NumPy arrays or have unsupported dtypes.
        ValueError: If dimensions, channels, values, or image content are invalid.
    """
    reference, enhanced, data_range, _ = _validate_pair(reference, enhanced)
    reference_float = reference.astype(np.float64, copy=False)
    enhanced_float = enhanced.astype(np.float64, copy=False)
    mean_squared_error = float(np.mean(np.square(reference_float - enhanced_float)))
    if mean_squared_error == 0:
        return math.inf
    return float(10.0 * math.log10((data_range * data_range) / mean_squared_error))


def calculate_ssim(reference: np.ndarray, enhanced: np.ndarray) -> float:
    """Calculate structural similarity (SSIM) for a reference image pair.

    SSIM compares local luminance, contrast, and structure; its score is
    typically in [0, 1] for nonnegative images, with 1 indicating identical
    images. It requires a reference because similarity is measured between
    corresponding regions of the enhanced and ground-truth images.

    Accepts matching 2D grayscale or channels-last RGB NumPy arrays. ``uint8``
    images use data range 255. Floating-point images must be finite and
    normalized to [0, 1], using data range 1.0. Small images use the largest
    supported odd SSIM window up to 7; spatial dimensions below 3 are rejected.

    Raises:
        TypeError: If inputs are not NumPy arrays or have unsupported dtypes.
        ValueError: If dimensions, channels, values, or image size are invalid.
    """
    reference, enhanced, data_range, channel_axis = _validate_pair(
        reference, enhanced
    )
    spatial_shape = reference.shape[:2]
    shortest_dimension = min(spatial_shape)
    window_size = min(7, shortest_dimension)
    if window_size % 2 == 0:
        window_size -= 1
    if window_size < 3:
        raise ValueError("SSIM requires image height and width of at least 3 pixels.")

    score = structural_similarity(
        reference,
        enhanced,
        data_range=data_range,
        channel_axis=channel_axis,
        win_size=window_size,
    )
    return float(score)


def evaluate_paired_image(
    reference: np.ndarray,
    enhanced: np.ndarray,
) -> dict[str, float]:
    """Return PSNR and SSIM for an enhanced image and its reference.

    Both values are computed from the supplied image pair; no metric values are
    inferred when a reference is missing. Input formats and value ranges match
    :func:`calculate_psnr` and :func:`calculate_ssim`.
    """
    return {
        "psnr": calculate_psnr(reference, enhanced),
        "ssim": calculate_ssim(reference, enhanced),
    }


def _get_lpips_model() -> tuple[Any, Any, Any]:
    """Load and cache the pretrained AlexNet LPIPS model on the available device."""
    global _LPIPS_MODEL
    with _LPIPS_MODEL_LOCK:
        if _LPIPS_MODEL is not None:
            return _LPIPS_MODEL

        try:
            import lpips
            import torch
        except ImportError as error:
            raise ImportError(
                "LPIPS evaluation requires the 'lpips' and 'torch' packages. "
                "Install project dependencies with `pip install -r requirements.txt`."
            ) from error

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = lpips.LPIPS(net="alex", verbose=False).to(device)
        model.eval()
        _LPIPS_MODEL = (torch, model, device)
        return _LPIPS_MODEL


def _lpips_tensor(image: np.ndarray, torch: Any, device: Any) -> Any:
    """Convert an HWC RGB image to a normalized NCHW LPIPS tensor."""
    image_float = image.astype(np.float32, copy=False)
    if image.dtype == np.uint8:
        image_float = image_float / 127.5 - 1.0
    else:
        image_float = image_float * 2.0 - 1.0
    image_float = np.ascontiguousarray(image_float.transpose(2, 0, 1)[None, ...])
    return torch.from_numpy(image_float).to(device=device, dtype=torch.float32)


def calculate_lpips(reference: np.ndarray, enhanced: np.ndarray) -> float:
    """Calculate LPIPS perceptual distance for an RGB image and its reference.

    LPIPS compares deep features from a pretrained AlexNet network; lower scores
    indicate greater perceptual similarity. It requires both a reference image
    and the enhanced image, and is not a no-reference quality score.

    Inputs must be matching, channels-last RGB NumPy arrays of at least
    64 x 64 pixels. ``uint8`` images are interpreted over [0, 255], while
    floating-point images must be finite and normalized to [0, 1]. Both are
    converted to the [-1, 1] tensor range expected by LPIPS. The pretrained
    model is loaded once and uses CUDA when available, otherwise CPU.

    Raises:
        TypeError: If inputs are not NumPy arrays or use unsupported dtypes.
        ValueError: If dimensions, channels, values, or spatial size are invalid.
        ImportError: If the ``lpips`` or ``torch`` dependency is unavailable.
    """
    if not isinstance(reference, np.ndarray) or not isinstance(enhanced, np.ndarray):
        raise TypeError("reference and enhanced must both be NumPy arrays.")
    if reference.ndim != 3 or enhanced.ndim != 3:
        raise ValueError("LPIPS inputs must be 3D channels-last RGB arrays.")
    if reference.shape != enhanced.shape:
        raise ValueError(
            "reference and enhanced must have identical dimensions; "
            f"got {reference.shape} and {enhanced.shape}."
        )
    if reference.shape[-1] != 3:
        raise ValueError("LPIPS inputs must have exactly three RGB channels.")
    if reference.size == 0 or any(dimension == 0 for dimension in reference.shape):
        raise ValueError("Images must not be empty.")
    if min(reference.shape[:2]) < 64:
        raise ValueError("LPIPS AlexNet inputs must be at least 64 x 64 pixels.")

    reference_is_uint8 = reference.dtype == np.uint8
    enhanced_is_uint8 = enhanced.dtype == np.uint8
    reference_is_float = np.issubdtype(reference.dtype, np.floating)
    enhanced_is_float = np.issubdtype(enhanced.dtype, np.floating)
    if reference_is_uint8 and enhanced_is_uint8:
        pass
    elif reference_is_float and enhanced_is_float:
        if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(enhanced)):
            raise ValueError("Floating-point images must contain only finite values.")
        if (
            np.any(reference < 0)
            or np.any(reference > 1)
            or np.any(enhanced < 0)
            or np.any(enhanced > 1)
        ):
            raise ValueError(
                "Floating-point image values must be normalized to the [0, 1] range."
            )
    else:
        raise TypeError(
            "Both images must use uint8 or both must use floating-point values "
            "normalized to [0, 1]."
        )

    torch, model, device = _get_lpips_model()
    reference_tensor = _lpips_tensor(reference, torch, device)
    enhanced_tensor = _lpips_tensor(enhanced, torch, device)
    with torch.no_grad():
        score = model(reference_tensor, enhanced_tensor)
    return float(score.item())


def _get_niqe_metric() -> tuple[Any, Any]:
    """Create and cache the pyiqa NIQE metric on the available device."""
    global _NIQE_METRIC
    with _NIQE_METRIC_LOCK:
        if _NIQE_METRIC is not None:
            return _NIQE_METRIC

        try:
            import pyiqa
            import torch
        except ImportError as error:
            raise ImportError(
                "NIQE evaluation requires the 'pyiqa' and 'torch' packages. "
                "Install project dependencies with `pip install -r requirements.txt`."
            ) from error

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        metric = pyiqa.create_metric("niqe", device=device)
        metric.eval()
        _NIQE_METRIC = (torch, metric)
        return _NIQE_METRIC


def calculate_niqe(image: np.ndarray) -> float:
    """Calculate no-reference NIQE for a single RGB image.

    NIQE measures deviations from natural-image statistics without requiring
    a reference image; lower scores generally indicate better perceptual
    quality. It is not a task-specific ground-truth metric.

    ``image`` must be a channels-last RGB NumPy array with at least 96 pixels
    in each spatial dimension. ``uint8`` values are interpreted over [0, 255];
    finite floating-point values must be normalized to [0, 1]. RGB is converted
    to the Y/luma plane using the standard 0.299/0.587/0.114 coefficients, then
    passed as a single-channel [0, 1] tensor to the established pyiqa NIQE
    implementation. The pyiqa pretrained pristine-image statistics are cached
    by that library; the metric instance is cached here and uses CUDA when
    available, otherwise CPU.

    Raises:
        TypeError: If the input is not a NumPy array or has an unsupported dtype.
        ValueError: If the image shape, channels, values, or dimensions are invalid.
        ImportError: If ``pyiqa`` or ``torch`` is unavailable.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array.")
    if image.ndim != 3:
        raise ValueError("NIQE input must be a 3D channels-last RGB image.")
    if image.shape[-1] != 3:
        raise ValueError("NIQE input must have exactly three RGB channels.")
    if image.size == 0 or any(dimension == 0 for dimension in image.shape):
        raise ValueError("Image must not be empty.")
    if min(image.shape[:2]) < 96:
        raise ValueError("NIQE requires image height and width of at least 96 pixels.")

    image_is_uint8 = image.dtype == np.uint8
    image_is_float = np.issubdtype(image.dtype, np.floating)
    if image_is_uint8:
        rgb = image.astype(np.float32) / 255.0
    elif image_is_float:
        if not np.all(np.isfinite(image)):
            raise ValueError("Floating-point images must contain only finite values.")
        if np.any(image < 0) or np.any(image > 1):
            raise ValueError(
                "Floating-point image values must be normalized to the [0, 1] range."
            )
        rgb = image.astype(np.float32, copy=False)
    else:
        raise TypeError(
            "Image must use uint8 or floating-point values normalized to [0, 1]."
        )

    luma = (
        0.299 * rgb[:, :, 0]
        + 0.587 * rgb[:, :, 1]
        + 0.114 * rgb[:, :, 2]
    )
    luma = np.ascontiguousarray(luma[None, None, :, :], dtype=np.float32)

    torch, metric = _get_niqe_metric()
    image_tensor = torch.from_numpy(luma).to(device=metric.device, dtype=torch.float32)
    with torch.no_grad():
        score = metric(image_tensor)
    score_value = float(score.item())
    if not math.isfinite(score_value):
        raise RuntimeError("The NIQE implementation returned a non-finite score.")
    return score_value
