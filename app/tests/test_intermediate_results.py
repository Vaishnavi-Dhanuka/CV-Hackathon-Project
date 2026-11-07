"""Tests for rendering actual restoration-pipeline artifacts."""

from contextlib import nullcontext
from unittest.mock import Mock

import numpy as np
from PIL import Image

from app.components import image_display


def _mock_streamlit(monkeypatch) -> dict[str, Mock]:
    calls = {
        name: Mock()
        for name in (
            "divider",
            "markdown",
            "subheader",
            "caption",
            "image",
            "info",
            "write",
            "error",
        )
    }
    for name, mock in calls.items():
        monkeypatch.setattr(image_display.st, name, mock)
    monkeypatch.setattr(
        image_display.st,
        "columns",
        lambda count, **_: [nullcontext() for _ in range(count)],
    )
    monkeypatch.setattr(image_display.st, "expander", lambda *_: nullcontext())
    monkeypatch.setattr(image_display.st, "container", lambda **_: nullcontext())
    return calls


def test_hybrid_intermediate_section_displays_real_branch_and_fusion_outputs(
    monkeypatch,
) -> None:
    calls = _mock_streamlit(monkeypatch)
    artifacts = {
        name: Image.fromarray(np.full((2, 3, 3), index, dtype=np.uint8))
        for index, name in enumerate(
            ("illumination", "reflectance", "retinex", "clahe", "denoised", "final"),
            start=1,
        )
    }
    classical_image = Image.new("RGB", (3, 2), "red")
    deep_learning_image = Image.new("RGB", (3, 2), "blue")
    hybrid_image = Image.new("RGB", (3, 2), "purple")
    metadata = {
        "latency_ms": {"preprocess": 2.5, "retinex": 5.0, "total": 7.5},
        "global_baseline_weights": {
            "classical_weight": 0.6,
            "dl_weight": 0.4,
        }
    }
    classical_metadata = {"latency_ms": {"preprocess": 2.5, "retinex": 5.0}}

    image_display.render_intermediate_visualizations(
        selected_mode="Hybrid Intelligence",
        classical_image=classical_image,
        classical_artifacts=artifacts,
        classical_metadata=classical_metadata,
        deep_learning_image=deep_learning_image,
        hybrid_image=hybrid_image,
        fusion_metadata=metadata,
    )

    displayed = [call.args[0] for call in calls["image"].call_args_list]
    for artifact in artifacts.values():
        assert any(image is artifact for image in displayed)
    assert any(image is classical_image for image in displayed)
    assert any(image is deep_learning_image for image in displayed)
    assert any(image is hybrid_image for image in displayed)
    assert any(
        "0.600" in str(call.args[0])
        for call in calls["write"].call_args_list
    )
    assert any(
        "Actual M1 stage timings" in str(call.args[0])
        for call in calls["write"].call_args_list
    )


def test_unavailable_stages_are_labeled_without_fabricated_images(monkeypatch) -> None:
    calls = _mock_streamlit(monkeypatch)
    deep_learning_image = Image.new("RGB", (3, 2), "blue")

    image_display.render_intermediate_visualizations(
        selected_mode="Official RetinexFormer",
        deep_learning_image=deep_learning_image,
    )

    calls["image"].assert_called_once_with(
        deep_learning_image,
        use_container_width=True,
    )
    messages = [str(call.args[0]) for call in calls["info"].call_args_list]
    assert any("Not run for the selected pipeline" in message for message in messages)
    assert any("Hybrid fusion has not run" in message for message in messages)


def test_comparison_cards_still_render_all_four_actual_images(monkeypatch) -> None:
    calls = _mock_streamlit(monkeypatch)
    images = [Image.new("RGB", (3, 2), color) for color in ("black", "red", "blue", "purple")]

    image_display.render_comparison_results(
        image=images[0],
        selected_mode="Hybrid Intelligence",
        classical_image=images[1],
        deep_learning_image=images[2],
        hybrid_image=images[3],
    )

    displayed = [call.args[0] for call in calls["image"].call_args_list]
    assert len(displayed) == 4
    for image in images:
        assert any(output is image for output in displayed)
