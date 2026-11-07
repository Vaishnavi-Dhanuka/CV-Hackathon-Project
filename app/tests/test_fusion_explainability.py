"""Tests for validation and rendering of Hybrid fusion contribution maps."""

import numpy as np
import pytest
from PIL import Image

from app.components.fusion_explainability import (
    average_spatial_contributions,
    build_contribution_figure,
    build_overlay_image,
    build_weight_map_figure,
    validate_fusion_metadata,
)


def _metadata() -> dict[str, object]:
    classical_map = np.array(
        [[0.7, 0.6], [0.4, 0.5]],
        dtype=np.float32,
    )
    return {
        "classical_weight_map": classical_map,
        "dl_weight_map": 1.0 - classical_map,
        "global_baseline_weights": {
            "classical_weight": 0.55,
            "dl_weight": 0.45,
        },
    }


def test_valid_maps_are_preserved_and_figures_use_weight_ranges() -> None:
    metadata = _metadata()
    classical_map, dl_map, classical_baseline, dl_baseline = (
        validate_fusion_metadata(metadata, expected_shape=(2, 2))
    )

    np.testing.assert_array_equal(classical_map, metadata["classical_weight_map"])
    np.testing.assert_array_equal(dl_map, metadata["dl_weight_map"])
    assert (classical_baseline, dl_baseline) == (0.55, 0.45)

    map_figure = build_weight_map_figure(classical_map, dl_map)
    assert len(map_figure.data) == 2
    assert map_figure.data[0].zmin == 0
    assert map_figure.data[0].zmax == 1
    assert map_figure.data[1].zmin == 0
    assert map_figure.data[1].zmax == 1

    contribution_figure = build_contribution_figure(classical_map, dl_map)
    contribution = np.asarray(contribution_figure.data[0].z)
    np.testing.assert_allclose(contribution, classical_map - dl_map)
    assert contribution_figure.data[0].zmin == -1
    assert contribution_figure.data[0].zmax == 1
    mean_classical, mean_dl = average_spatial_contributions(classical_map, dl_map)
    assert mean_classical + mean_dl == pytest.approx(1.0)
    assert mean_classical == pytest.approx(float(np.mean(classical_map)))
    assert mean_dl == pytest.approx(float(np.mean(dl_map)))


def test_average_spatial_contributions_rejects_noncomplementary_maps() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        average_spatial_contributions(
            np.full((2, 2), 0.6, dtype=np.float32),
            np.full((2, 2), 0.6, dtype=np.float32),
        )


@pytest.mark.parametrize(
    ("metadata", "expected_shape", "message"),
    [
        (None, None, "metadata is unavailable"),
        ({}, None, "both NumPy weight maps"),
        (
            {
                **_metadata(),
                "dl_weight_map": np.full((2, 2), 0.3, dtype=np.float32),
            },
            None,
            "must sum to one per pixel",
        ),
        (
            {
                **_metadata(),
                "classical_weight_map": np.full((2, 2), 1.2, dtype=np.float32),
                "dl_weight_map": np.full((2, 2), -0.2, dtype=np.float32),
            },
            None,
            r"normalized to \[0, 1\]",
        ),
        (
            {
                **_metadata(),
                "classical_weight_map": np.full((3, 2), 0.5, dtype=np.float32),
                "dl_weight_map": np.full((3, 2), 0.5, dtype=np.float32),
            },
            (2, 2),
            "must match the original image",
        ),
        (
            {
                **_metadata(),
                "global_baseline_weights": {
                    "classical_weight": 0.7,
                    "dl_weight": 0.4,
                },
            },
            None,
            "baseline weights must sum to one",
        ),
    ],
)
def test_invalid_or_missing_metadata_is_rejected(
    metadata: dict[str, object] | None,
    expected_shape: tuple[int, int] | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_fusion_metadata(metadata, expected_shape=expected_shape)


def test_overlay_preserves_dimensions_and_tints_contribution_directions() -> None:
    image = Image.new("RGB", (2, 2), (100, 100, 100))
    classical_map, dl_map, _, _ = validate_fusion_metadata(_metadata())

    overlay = build_overlay_image(image, classical_map, dl_map)

    assert overlay.shape == (2, 2, 3)
    assert overlay.dtype == np.uint8
    assert overlay.flags.c_contiguous
    assert overlay[0, 0, 0] > overlay[0, 0, 2]
    assert overlay[1, 0, 2] > overlay[1, 0, 0]
