"""Tests for the adapter to the M1 classical enhancement package."""

from unittest.mock import patch

import cv2
import numpy as np
import pytest
from PIL import Image

from hybrid.classical_client import (
    enhance_with_classical,
    enhance_with_classical_artifacts,
)


_ARTIFACT_NAMES = (
    "illumination",
    "reflectance",
    "retinex",
    "clahe",
    "denoised",
    "final",
)


def _pipeline_result(image: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    artifacts = {
        name: np.full_like(image, index * 10)
        for index, name in enumerate(_ARTIFACT_NAMES, start=1)
    }
    return artifacts["final"], {**artifacts, "meta": {"stage": "complete"}}


def test_legacy_adapter_contract_returns_full_pipeline_final() -> None:
    input_array = np.full((3, 4, 3), (10, 20, 30), dtype=np.uint8)
    enhanced = np.full((3, 4, 3), (40, 50, 60), dtype=np.uint8)

    with patch(
        "hybrid.classical_client.enhance_classical",
        return_value=(
            enhanced,
            {**_pipeline_result(input_array)[1], "final": enhanced},
        ),
    ) as pipeline:
        result, latency = enhance_with_classical(Image.fromarray(input_array))

    pipeline.assert_called_once()
    np.testing.assert_array_equal(pipeline.call_args.args[0], input_array)
    np.testing.assert_array_equal(result, enhanced)
    assert result.dtype == np.uint8
    assert result.flags.c_contiguous
    assert latency >= 0


def test_artifact_adapter_returns_actual_pipeline_stages_and_metadata() -> None:
    input_array = np.full((3, 4, 3), (10, 20, 30), dtype=np.uint8)
    final, intermediate = _pipeline_result(input_array)

    with patch(
        "hybrid.classical_client.enhance_classical",
        return_value=(final, intermediate),
    ) as pipeline:
        result, latency, artifacts, metadata = enhance_with_classical_artifacts(
            Image.fromarray(input_array)
        )

    pipeline.assert_called_once()
    np.testing.assert_array_equal(result, final)
    assert set(artifacts) == set(_ARTIFACT_NAMES)
    for name in _ARTIFACT_NAMES:
        np.testing.assert_array_equal(artifacts[name], intermediate[name])
    assert metadata == {"stage": "complete"}
    assert latency >= 0


def test_input_is_converted_to_rgb_before_full_pipeline() -> None:
    rgba_image = Image.new("RGBA", (4, 3), (10, 20, 30, 40))
    expected_rgb = np.full((3, 4, 3), (10, 20, 30), dtype=np.uint8)
    final, intermediate = _pipeline_result(expected_rgb)

    with patch(
        "hybrid.classical_client.enhance_classical",
        return_value=(final, intermediate),
    ) as pipeline:
        enhance_with_classical_artifacts(rgba_image)

    np.testing.assert_array_equal(pipeline.call_args.args[0], expected_rgb)
    assert pipeline.call_args.args[0].dtype == np.uint8


def test_downsized_pipeline_images_are_resized_to_original_size() -> None:
    original = Image.new("RGB", (8, 6), (10, 20, 30))
    small = np.full((3, 4, 3), 70, dtype=np.uint8)
    final, intermediate = _pipeline_result(small)

    with patch(
        "hybrid.classical_client.enhance_classical",
        return_value=(final, intermediate),
    ):
        result, _, artifacts, _ = enhance_with_classical_artifacts(original)

    expected = cv2.resize(final, (8, 6), interpolation=cv2.INTER_LINEAR)
    np.testing.assert_array_equal(result, expected)
    for name, artifact in artifacts.items():
        expected_artifact = cv2.resize(
            intermediate[name],
            (8, 6),
            interpolation=cv2.INTER_LINEAR,
        )
        np.testing.assert_array_equal(artifact, expected_artifact)


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


@pytest.mark.parametrize(
    "error", [ValueError("pipeline failed"), TypeError("bad input")]
)
def test_pipeline_errors_are_propagated(error: Exception) -> None:
    with patch(
        "hybrid.classical_client.enhance_classical",
        side_effect=error,
    ):
        with pytest.raises(type(error), match=str(error)):
            enhance_with_classical(Image.new("RGB", (4, 3)))


def test_invalid_pipeline_stage_is_rejected() -> None:
    image = Image.new("RGB", (4, 3))
    final, intermediate = _pipeline_result(np.asarray(image))
    intermediate["retinex"] = np.zeros((3, 4), dtype=np.uint8)

    with patch(
        "hybrid.classical_client.enhance_classical",
        return_value=(final, intermediate),
    ):
        with pytest.raises(ValueError, match="HxWx3 RGB"):
            enhance_with_classical_artifacts(image)
