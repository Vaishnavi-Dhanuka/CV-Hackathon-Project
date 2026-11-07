"""Focused tests for Deep Learning result caching and metric failures."""

from contextlib import nullcontext
from io import BytesIO
from unittest.mock import Mock

import numpy as np
from PIL import Image

from app import main as lumina_main


def _classical_result(image: np.ndarray, latency: float):
    artifacts = {
        name: image.copy()
        for name in ("illumination", "reflectance", "retinex", "clahe", "denoised", "final")
    }
    return image, latency, artifacts, {"pipeline": "M1"}


def test_deep_learning_result_is_cached_by_image_and_mode(monkeypatch) -> None:
    image = Image.new("RGB", (4, 3), (10, 20, 30))
    uploaded_bytes = b"same uploaded image"
    enhanced_array = np.full((3, 4, 3), 90, dtype=np.uint8)
    enhance = Mock(return_value=(enhanced_array, 1.25))
    niqe = Mock(return_value=4.5)
    noise_reduction = Mock(return_value=2.0)
    monkeypatch.setattr(lumina_main, "enhance_with_api", enhance)
    monkeypatch.setattr(lumina_main, "calculate_niqe", niqe)
    monkeypatch.setattr(lumina_main, "calculate_noise_reduction", noise_reduction)
    monkeypatch.setattr(lumina_main.st, "spinner", lambda _: nullcontext())
    session_state = {}

    first = lumina_main._get_deep_learning_result(
        image, uploaded_bytes, "Deep Learning", session_state
    )
    second = lumina_main._get_deep_learning_result(
        image, uploaded_bytes, "Deep Learning", session_state
    )

    assert first is second
    enhance.assert_called_once_with(image)
    niqe.assert_called_once_with(enhanced_array)
    noise_reduction.assert_called_once()
    assert first["latency_seconds"] == 1.25
    assert first["niqe"] == 4.5
    assert first["noise_reduction"] == 2.0
    assert first["error"] is None

    with Image.open(BytesIO(first["enhanced_png"])) as result_image:
        assert result_image.mode == "RGB"
        assert result_image.size == image.size
        np.testing.assert_array_equal(
            np.asarray(result_image),
            np.full((3, 4, 3), 90, dtype=np.uint8),
        )


def test_metric_failures_are_cached_without_failing_enhancement(monkeypatch) -> None:
    image = Image.new("RGB", (4, 3))
    monkeypatch.setattr(
        lumina_main,
        "enhance_with_api",
        Mock(return_value=(np.zeros((3, 4, 3), dtype=np.uint8), 0.5)),
    )
    monkeypatch.setattr(
        lumina_main,
        "calculate_niqe",
        Mock(side_effect=ValueError("image too small")),
    )
    monkeypatch.setattr(
        lumina_main,
        "calculate_noise_reduction",
        Mock(side_effect=RuntimeError("noise metric failed")),
    )
    monkeypatch.setattr(lumina_main.st, "spinner", lambda _: nullcontext())

    result = lumina_main._get_deep_learning_result(
        image, b"metrics fail", "Deep Learning", {}
    )

    assert result["enhanced_png"] is not None
    assert result["error"] is None
    assert result["niqe"] is None
    assert "image too small" in result["niqe_error"]
    assert result["noise_reduction"] is None
    assert "noise metric failed" in result["noise_reduction_error"]


def test_api_connection_error_is_cached_with_actionable_message(monkeypatch) -> None:
    image = Image.new("RGB", (4, 3))
    enhance = Mock(side_effect=ConnectionError("offline"))
    monkeypatch.setattr(lumina_main, "enhance_with_api", enhance)
    monkeypatch.setattr(lumina_main.st, "spinner", lambda _: nullcontext())
    session_state = {}

    first = lumina_main._get_deep_learning_result(
        image, b"api unavailable", "Deep Learning", session_state
    )
    second = lumina_main._get_deep_learning_result(
        image, b"api unavailable", "Deep Learning", session_state
    )

    enhance.assert_called_once_with(image)
    assert first is second
    assert "M2 server is running" in first["error"]
    assert "offline" in first["error"]
    assert first["enhanced_png"] is None


def test_classical_result_is_cached_and_computes_metrics(monkeypatch) -> None:
    image = Image.new("RGB", (4, 3), (10, 20, 30))
    enhanced_array = np.full((3, 4, 3), 90, dtype=np.uint8)
    enhance = Mock(return_value=_classical_result(enhanced_array, 0.25))
    niqe = Mock(return_value=4.0)
    noise = Mock(return_value=2.0)
    monkeypatch.setattr(lumina_main, "enhance_with_classical_artifacts", enhance)
    monkeypatch.setattr(lumina_main, "calculate_niqe", niqe)
    monkeypatch.setattr(lumina_main, "calculate_noise_reduction", noise)
    monkeypatch.setattr(lumina_main.st, "spinner", lambda _: nullcontext())
    state = {}

    first = lumina_main._get_classical_result(
        image, b"classical image", "Classical CV", state
    )
    second = lumina_main._get_classical_result(
        image, b"classical image", "Classical CV", state
    )

    assert first is second
    enhance.assert_called_once_with(image)
    niqe.assert_called_once_with(enhanced_array)
    noise.assert_called_once()
    assert first["latency_seconds"] == 0.25
    assert first["niqe"] == 4.0
    assert first["noise_reduction"] == 2.0
    assert first["pipeline_metadata"] == {"pipeline": "M1"}
    assert set(first["artifacts_png"]) == {
        "illumination",
        "reflectance",
        "retinex",
        "clahe",
        "denoised",
    }
    with Image.open(BytesIO(first["enhanced_png"])) as output:
        assert output.mode == "RGB"
        assert output.size == image.size


def test_switching_to_hybrid_reuses_cached_classical_and_dl_branches(monkeypatch):
    image = Image.new("RGB", (4, 3), (10, 20, 30))
    uploaded_bytes = b"switch between modes"
    classical_output = np.full((3, 4, 3), 50, dtype=np.uint8)
    dl_output = np.full((3, 4, 3), 90, dtype=np.uint8)
    classical = Mock(return_value=_classical_result(classical_output, 0.2))
    dl = Mock(return_value=(dl_output, 0.3))
    fusion = Mock(
        return_value=(
            np.full((3, 4, 3), 70, dtype=np.uint8),
            {
                "classical_weight_map": np.full((3, 4), 0.5, dtype=np.float32),
                "dl_weight_map": np.full((3, 4), 0.5, dtype=np.float32),
                "global_baseline_weights": {
                    "classical_weight": 0.5,
                    "dl_weight": 0.5,
                },
            },
        )
    )
    monkeypatch.setattr(lumina_main, "enhance_with_classical_artifacts", classical)
    monkeypatch.setattr(lumina_main, "enhance_with_api", dl)
    monkeypatch.setattr(lumina_main, "spatial_adaptive_fusion", fusion)
    monkeypatch.setattr(lumina_main, "calculate_niqe", Mock(return_value=5.0))
    monkeypatch.setattr(
        lumina_main,
        "calculate_noise_reduction",
        Mock(return_value=0.0),
    )
    monkeypatch.setattr(lumina_main.st, "spinner", lambda _: nullcontext())
    state = {}

    lumina_main._get_classical_result(
        image, uploaded_bytes, "Classical CV", state
    )
    lumina_main._get_deep_learning_result(
        image, uploaded_bytes, "Official RetinexFormer", state
    )
    hybrid = lumina_main._get_hybrid_result(
        image, uploaded_bytes, "Hybrid Intelligence", state
    )

    classical.assert_called_once_with(image)
    dl.assert_called_once_with(image)
    fusion.assert_called_once()
    assert hybrid["error"] is None
    assert hybrid["hybrid_png"] is not None
