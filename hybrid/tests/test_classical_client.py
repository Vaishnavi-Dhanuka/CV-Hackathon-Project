"""Tests for the adapter to the M1 classical enhancement package."""

from unittest.mock import patch

import cv2
import numpy as np
import pytest
from PIL import Image

from hybrid.classical_client import enhance_with_classical


def test_successful_enhancement_returns_contiguous_rgb_uint8_image() -> None:
    input_array = np.full((3, 4, 3), (10, 20, 30), dtype=np.uint8)
    enhanced = np.full((3, 4, 3), (40, 50, 60), dtype=np.uint8)

    with (
        patch(
            "hybrid.classical_client.prepare_image",
            return_value=(input_array, {}),
        ) as prepare,
        patch(
            "hybrid.classical_client.retinex_enhance",
            return_value=enhanced,
        ) as retinex,
    ):
        result, latency = enhance_with_classical(Image.fromarray(input_array))

    prepare.assert_called_once()
    retinex.assert_called_once_with(input_array)
    assert result.shape == (3, 4, 3)
    assert result.dtype == np.uint8
    assert result.flags.c_contiguous
    np.testing.assert_array_equal(result, enhanced)
    assert latency >= 0


def test_input_is_converted_to_rgb_before_preprocessing() -> None:
    rgba_image = Image.new("RGBA", (4, 3), (10, 20, 30, 40))
    expected_rgb = np.full((3, 4, 3), (10, 20, 30), dtype=np.uint8)

    with (
        patch(
            "hybrid.classical_client.prepare_image",
            return_value=(expected_rgb, {}),
        ) as prepare,
        patch(
            "hybrid.classical_client.retinex_enhance",
            return_value=expected_rgb,
        ),
    ):
        enhance_with_classical(rgba_image)

    np.testing.assert_array_equal(prepare.call_args.args[0], expected_rgb)
    assert prepare.call_args.args[0].dtype == np.uint8


def test_downsized_enhancement_is_resized_to_original_size() -> None:
    original = Image.new("RGB", (8, 6), (10, 20, 30))
    prepared = np.full((3, 4, 3), (10, 20, 30), dtype=np.uint8)
    enhanced = np.full((3, 4, 3), (70, 80, 90), dtype=np.uint8)

    with (
        patch(
            "hybrid.classical_client.prepare_image",
            return_value=(prepared, {"scale": 0.5}),
        ),
        patch(
            "hybrid.classical_client.retinex_enhance",
            return_value=enhanced,
        ),
    ):
        result, _ = enhance_with_classical(original)

    assert result.shape == (6, 8, 3)
    assert result.dtype == np.uint8
    assert result.flags.c_contiguous
    np.testing.assert_array_equal(
        result,
        cv2.resize(enhanced, (8, 6), interpolation=cv2.INTER_LINEAR),
    )


@pytest.mark.parametrize(
    ("image", "error", "message"),
    [
        (None, TypeError, "PIL.Image.Image"),
        (np.zeros((2, 2, 3), dtype=np.uint8), TypeError, "PIL.Image.Image"),
        (Image.Image(), ValueError, "non-empty dimensions"),
    ],
)
def test_invalid_inputs_are_rejected(
    image: object, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        enhance_with_classical(image)  # type: ignore[arg-type]


@pytest.mark.parametrize("error", [ValueError("invalid input"), TypeError("bad type")])
def test_preprocessing_errors_are_propagated(error: Exception) -> None:
    with patch(
        "hybrid.classical_client.prepare_image",
        side_effect=error,
    ):
        with pytest.raises(type(error), match=str(error)):
            enhance_with_classical(Image.new("RGB", (4, 3)))


@pytest.mark.parametrize("error", [ValueError("enhancement failed"), TypeError("bad data")])
def test_retinex_errors_are_propagated(error: Exception) -> None:
    prepared = np.zeros((3, 4, 3), dtype=np.uint8)
    with (
        patch(
            "hybrid.classical_client.prepare_image",
            return_value=(prepared, {}),
        ),
        patch(
            "hybrid.classical_client.retinex_enhance",
            side_effect=error,
        ),
    ):
        with pytest.raises(type(error), match=str(error)):
            enhance_with_classical(Image.new("RGB", (4, 3)))
