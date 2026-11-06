"""Tests for the model-agnostic deep-learning API client."""

from io import BytesIO
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import requests
from PIL import Image

from hybrid.dl_client import enhance_with_api


def _image_bytes(
    image_format: str = "PNG",
    size: tuple[int, int] = (3, 2),
    color: tuple[int, int, int] = (20, 40, 60),
) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", size, color).save(buffer, format=image_format)
    return buffer.getvalue()


def _response(
    content: bytes,
    *,
    content_type: str = "image/png",
    ok: bool = True,
    status_code: int = 200,
    reason: str = "OK",
) -> SimpleNamespace:
    return SimpleNamespace(
        ok=ok,
        status_code=status_code,
        reason=reason,
        headers={"Content-Type": content_type},
        content=content,
    )


def test_successful_png_response_returns_rgb_uint8_array_and_latency() -> None:
    input_image = Image.new("RGB", (3, 2), (1, 2, 3))
    expected_pixel = (20, 40, 60)
    response = _response(_image_bytes(color=expected_pixel))

    with patch("hybrid.dl_client.requests.post", return_value=response):
        enhanced, latency = enhance_with_api(input_image)

    assert isinstance(enhanced, np.ndarray)
    assert enhanced.shape == (2, 3, 3)
    assert enhanced.dtype == np.uint8
    np.testing.assert_array_equal(
        enhanced, np.full((2, 3, 3), expected_pixel, dtype=np.uint8)
    )
    assert latency >= 0


def test_input_is_converted_to_rgb_before_upload() -> None:
    input_image = Image.new("RGBA", (3, 2), (10, 20, 30, 40))
    response = _response(_image_bytes())

    with patch("hybrid.dl_client.requests.post", return_value=response) as post:
        enhance_with_api(input_image)

    uploaded_bytes = post.call_args.kwargs["files"]["file"][1]
    with Image.open(BytesIO(uploaded_bytes)) as uploaded_image:
        assert uploaded_image.format == "PNG"
        assert uploaded_image.mode == "RGB"
        assert uploaded_image.getpixel((0, 0)) == (10, 20, 30)


def test_upload_uses_expected_multipart_field_filename_and_mime_type() -> None:
    response = _response(_image_bytes())

    with patch("hybrid.dl_client.requests.post", return_value=response) as post:
        enhance_with_api(Image.new("RGB", (3, 2)))

    files = post.call_args.kwargs["files"]
    assert set(files) == {"file"}
    filename, content, mime_type = files["file"]
    assert filename == "image.png"
    assert content
    assert mime_type == "image/png"


def test_non_pil_input_raises_type_error() -> None:
    with pytest.raises(TypeError, match="PIL.Image.Image"):
        enhance_with_api(np.zeros((2, 3, 3), dtype=np.uint8))  # type: ignore[arg-type]


def test_empty_image_dimensions_raise_value_error() -> None:
    with pytest.raises(ValueError, match="non-empty dimensions"):
        enhance_with_api(Image.Image())


def test_connection_error_raises_connection_error() -> None:
    with patch(
        "hybrid.dl_client.requests.post",
        side_effect=requests.exceptions.ConnectionError("offline"),
    ):
        with pytest.raises(ConnectionError, match="Could not connect"):
            enhance_with_api(Image.new("RGB", (3, 2)))


def test_timeout_raises_timeout_error() -> None:
    with patch(
        "hybrid.dl_client.requests.post",
        side_effect=requests.exceptions.Timeout("slow"),
    ):
        with pytest.raises(TimeoutError, match="Timed out"):
            enhance_with_api(Image.new("RGB", (3, 2)))


def test_other_request_exception_raises_runtime_error() -> None:
    with patch(
        "hybrid.dl_client.requests.post",
        side_effect=requests.exceptions.RequestException("request failed"),
    ):
        with pytest.raises(RuntimeError, match="request failed"):
            enhance_with_api(Image.new("RGB", (3, 2)))


def test_non_success_http_response_raises_runtime_error() -> None:
    response = _response(
        b"server error",
        content_type="text/plain",
        ok=False,
        status_code=500,
        reason="Internal Server Error",
    )

    with patch("hybrid.dl_client.requests.post", return_value=response):
        with pytest.raises(RuntimeError, match="500 Internal Server Error"):
            enhance_with_api(Image.new("RGB", (3, 2)))


def test_unexpected_content_type_raises_runtime_error() -> None:
    response = _response(_image_bytes(), content_type="application/json")

    with patch("hybrid.dl_client.requests.post", return_value=response):
        with pytest.raises(RuntimeError, match="unexpected content type"):
            enhance_with_api(Image.new("RGB", (3, 2)))


def test_invalid_png_bytes_raise_runtime_error() -> None:
    response = _response(b"not an image")

    with patch("hybrid.dl_client.requests.post", return_value=response):
        with pytest.raises(RuntimeError, match="invalid PNG"):
            enhance_with_api(Image.new("RGB", (3, 2)))


def test_output_dimensions_different_from_input_raise_value_error() -> None:
    response = _response(_image_bytes(size=(4, 2)))

    with patch("hybrid.dl_client.requests.post", return_value=response):
        with pytest.raises(ValueError, match="dimensions"):
            enhance_with_api(Image.new("RGB", (3, 2)))


def test_png_content_type_with_non_png_image_format_is_rejected() -> None:
    response = _response(
        _image_bytes(image_format="JPEG"),
        content_type="image/png",
    )

    with patch("hybrid.dl_client.requests.post", return_value=response):
        with pytest.raises(RuntimeError, match="declared image/png"):
            enhance_with_api(Image.new("RGB", (3, 2)))
