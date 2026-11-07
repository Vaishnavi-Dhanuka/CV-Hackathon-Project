"""Tests for the Streamlit Hybrid Intelligence execution workflow."""

from contextlib import nullcontext
from unittest.mock import Mock

import numpy as np
from PIL import Image

from app import main as lumina_main


def _classical_result(image: np.ndarray, latency: float):
    artifacts = {
        name: image.copy()
        for name in (
            "illumination",
            "reflectance",
            "retinex",
            "clahe",
            "denoised",
            "final",
        )
    }
    return image, latency, artifacts, {"pipeline": "M1"}


def test_hybrid_result_passes_both_outputs_and_original_to_fusion(monkeypatch) -> None:
    image = Image.new("RGB", (4, 3), (10, 20, 30))
    input_array = np.asarray(image).copy()
    classical_array = np.full((3, 4, 3), 40, dtype=np.uint8)
    dl_array = np.full((3, 4, 3), 80, dtype=np.uint8)
    fused_array = np.full((3, 4, 3), 60, dtype=np.uint8)
    fusion_metadata = {
        "classical_weight_map": np.full((3, 4), 0.6, dtype=np.float32),
        "dl_weight_map": np.full((3, 4), 0.4, dtype=np.float32),
        "global_baseline_weights": {
            "classical_weight": 0.55,
            "dl_weight": 0.45,
        },
    }
    classical = Mock(return_value=_classical_result(classical_array, 0.4))
    dl = Mock(return_value=(dl_array, 1.2))
    fusion = Mock(return_value=(fused_array, fusion_metadata))
    monkeypatch.setattr(lumina_main, "enhance_with_classical_artifacts", classical)
    monkeypatch.setattr(lumina_main, "enhance_with_api", dl)
    monkeypatch.setattr(lumina_main, "spatial_adaptive_fusion", fusion)
    monkeypatch.setattr(lumina_main, "calculate_niqe", Mock(return_value=4.2))
    monkeypatch.setattr(
        lumina_main,
        "calculate_noise_reduction",
        Mock(return_value=1.5),
    )
    monkeypatch.setattr(lumina_main.st, "spinner", lambda _: nullcontext())
    session_state = {}

    result = lumina_main._get_hybrid_result(
        image,
        b"uploaded hybrid input",
        "Hybrid Intelligence",
        session_state,
    )

    classical.assert_called_once_with(image)
    dl.assert_called_once_with(image)
    fusion.assert_called_once()
    args = fusion.call_args.args
    np.testing.assert_array_equal(args[0], classical_array)
    np.testing.assert_array_equal(args[1], dl_array)
    np.testing.assert_array_equal(args[2], input_array)
    assert fusion.call_args.kwargs["return_weights"] is True
    assert result["classical_latency_seconds"] == 0.4
    assert result["dl_latency_seconds"] == 1.2
    assert result["hybrid_png"] is not None
    assert result["classical_artifacts_png"] is not None
    assert result["classical_pipeline_metadata"] == {"pipeline": "M1"}
    assert result["fusion_metadata"] is fusion_metadata
    assert result["error"] is None


def test_hybrid_result_is_cached_by_image_content_and_mode(monkeypatch) -> None:
    image = Image.new("RGB", (4, 3))
    enhanced_array = np.zeros((3, 4, 3), dtype=np.uint8)
    classical = Mock(return_value=_classical_result(enhanced_array, 0.2))
    dl = Mock(return_value=(enhanced_array, 0.3))
    fusion = Mock(
        return_value=(
            enhanced_array,
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
    session_state = {}

    first = lumina_main._get_hybrid_result(
        image, b"cached image", "Hybrid Intelligence", session_state
    )
    second = lumina_main._get_hybrid_result(
        image, b"cached image", "Hybrid Intelligence", session_state
    )

    assert first is second
    classical.assert_called_once_with(image)
    dl.assert_called_once_with(image)
    fusion.assert_called_once()


def test_hybrid_cache_does_not_reuse_result_across_modes(monkeypatch) -> None:
    image = Image.new("RGB", (4, 3))
    enhanced_array = np.zeros((3, 4, 3), dtype=np.uint8)
    classical = Mock(return_value=_classical_result(enhanced_array, 0.2))
    dl = Mock(return_value=(enhanced_array, 0.3))
    fusion = Mock(
        return_value=(
            enhanced_array,
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
    session_state = {}

    first = lumina_main._get_hybrid_result(
        image, b"same content", "Hybrid Intelligence", session_state
    )
    second = lumina_main._get_hybrid_result(
        image, b"same content", "different mode", session_state
    )

    assert first is not second
    assert first["key"] != second["key"]
    assert classical.call_count == 1
    assert dl.call_count == 1
    assert fusion.call_count == 2


def test_failed_enhancement_branch_does_not_run_fusion_or_create_result(
    monkeypatch,
) -> None:
    image = Image.new("RGB", (4, 3))
    enhanced_array = np.zeros((3, 4, 3), dtype=np.uint8)
    monkeypatch.setattr(
        lumina_main,
        "enhance_with_classical_artifacts",
        Mock(return_value=_classical_result(enhanced_array, 0.2)),
    )
    monkeypatch.setattr(
        lumina_main,
        "enhance_with_api",
        Mock(side_effect=ConnectionError("API offline")),
    )
    fusion = Mock()
    monkeypatch.setattr(lumina_main, "spatial_adaptive_fusion", fusion)
    monkeypatch.setattr(lumina_main.st, "spinner", lambda _: nullcontext())

    result = lumina_main._get_hybrid_result(
        image, b"failed branch", "Hybrid Intelligence", {}
    )

    fusion.assert_not_called()
    assert result["hybrid_png"] is None
    assert result["dl_error"] is not None
    assert "API offline" in result["dl_error"]
    assert "branch failed" in result["error"]


def test_fusion_failure_does_not_create_hybrid_result(monkeypatch) -> None:
    image = Image.new("RGB", (4, 3))
    enhanced_array = np.zeros((3, 4, 3), dtype=np.uint8)
    monkeypatch.setattr(
        lumina_main,
        "enhance_with_classical_artifacts",
        Mock(return_value=_classical_result(enhanced_array, 0.2)),
    )
    monkeypatch.setattr(
        lumina_main,
        "enhance_with_api",
        Mock(return_value=(enhanced_array, 0.3)),
    )
    monkeypatch.setattr(
        lumina_main,
        "spatial_adaptive_fusion",
        Mock(side_effect=ValueError("incompatible results")),
    )
    monkeypatch.setattr(lumina_main.st, "spinner", lambda _: nullcontext())

    result = lumina_main._get_hybrid_result(
        image, b"fusion failure", "Hybrid Intelligence", {}
    )

    assert result["hybrid_png"] is None
    assert "Hybrid fusion failed" in result["error"]
    assert "incompatible results" in result["error"]
