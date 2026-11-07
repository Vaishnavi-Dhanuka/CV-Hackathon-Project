"""Tests for user-facing evaluation metric labels and explanations."""

from unittest.mock import Mock

from app.components import metrics_display


def _mock_streamlit(monkeypatch) -> tuple[list[Mock], Mock]:
    metric_calls: list[Mock] = []
    caption = Mock()

    def columns(count: int) -> list[Mock]:
        result = []
        for _ in range(count):
            column = Mock()
            column.metric = Mock()
            metric_calls.append(column.metric)
            result.append(column)
        return result

    monkeypatch.setattr(metrics_display.st, "divider", Mock())
    monkeypatch.setattr(metrics_display.st, "markdown", Mock())
    monkeypatch.setattr(metrics_display.st, "subheader", Mock())
    monkeypatch.setattr(metrics_display.st, "caption", caption)
    monkeypatch.setattr(metrics_display.st, "columns", columns)
    return metric_calls, caption


def test_reference_metrics_and_niqe_dependency_have_user_friendly_display(
    monkeypatch,
) -> None:
    metric_calls, caption = _mock_streamlit(monkeypatch)

    metrics_display.render_metrics(
        niqe=None,
        niqe_error=(
            "ImportError: NIQE evaluation requires the 'pyiqa' and 'torch' packages."
        ),
        noise_reduction=1.25,
        latency_seconds=0.42,
        latency_label="Processing time",
    )

    metrics = [call.call_args.args[:2] for call in metric_calls]
    assert ("PSNR", "N/A") in metrics
    assert ("SSIM", "N/A") in metrics
    assert ("LPIPS", "N/A") in metrics
    assert ("NIQE", "N/A") in metrics
    assert ("Processing time", "0.42 s") in metrics
    assert ("Noise reduction (proxy)", "1.25") in metrics
    assert ("Edge preservation", "N/A") in metrics

    captions = [call.args[0] for call in caption.call_args_list]
    assert (
        "Reference-based metrics require a corresponding clean ground-truth image."
        in captions
    )
    assert (
        "NIQE is currently unavailable because its evaluation dependency is not installed."
        in captions
    )
    assert not any("ImportError" in message for message in captions)

    noise_help = next(
        call.call_args.kwargs["help"]
        for call in metric_calls
        if call.call_args.args[0] == "Noise reduction (proxy)"
    )
    assert "not a measure of restoration quality" in noise_help
    edge_help = next(
        call.call_args.kwargs["help"]
        for call in metric_calls
        if call.call_args.args[0] == "Edge preservation"
    )
    assert "ground-truth reference image" in edge_help


def test_non_dependency_niqe_error_remains_visible(monkeypatch) -> None:
    _, caption = _mock_streamlit(monkeypatch)

    metrics_display.render_metrics(niqe_error="ValueError: invalid image")

    captions = [call.args[0] for call in caption.call_args_list]
    assert "NIQE unavailable: ValueError: invalid image" in captions
