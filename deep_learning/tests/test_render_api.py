import asyncio
from io import BytesIO

from fastapi import HTTPException
from fastapi.testclient import TestClient
from PIL import Image
from starlette.datastructures import UploadFile

from deep_learning import api


def _upload_image(width: int = 2, height: int = 2) -> UploadFile:
    content = BytesIO()
    Image.new("RGB", (width, height), "black").save(content, format="PNG")
    content.seek(0)
    return UploadFile(filename="image.png", file=content)


def test_health_reports_missing_model_assets(monkeypatch, tmp_path):
    monkeypatch.setattr(api.retinex_inference, "CONFIG", tmp_path / "missing.yml")
    monkeypatch.setattr(api.retinex_inference, "CHECKPOINT", tmp_path / "missing.pth")

    try:
        api.health()
    except HTTPException as error:
        assert error.status_code == 503
        assert error.detail == "Model assets are not ready."
    else:
        raise AssertionError("health check accepted missing model assets")


def test_health_reports_ready_model_assets(monkeypatch, tmp_path):
    config = tmp_path / "model.yml"
    checkpoint = tmp_path / "model.pth"
    config.touch()
    checkpoint.touch()
    monkeypatch.setattr(api.retinex_inference, "CONFIG", config)
    monkeypatch.setattr(api.retinex_inference, "CHECKPOINT", checkpoint)

    assert api.health() == {"status": "ok"}


def test_enhance_rejects_upload_over_configured_byte_limit(monkeypatch):
    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 8)

    try:
        asyncio.run(api.enhance(_upload_image()))
    except HTTPException as error:
        assert error.status_code == 413
    else:
        raise AssertionError("oversized upload was accepted")


def test_http_request_limit_rejects_large_multipart_body_early(monkeypatch):
    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 16)

    response = TestClient(api.app).post(
        "/enhance",
        content=b"x" * 70000,
        headers={"content-type": "application/octet-stream"},
    )

    assert response.status_code == 413
    assert response.json()["detail"] == "Image upload exceeds the configured size limit."


def test_enhance_rejects_image_over_configured_pixel_limit(monkeypatch):
    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", None)
    monkeypatch.setattr(api, "MAX_IMAGE_PIXELS", 3)

    try:
        asyncio.run(api.enhance(_upload_image(2, 2)))
    except HTTPException as error:
        assert error.status_code == 413
    else:
        raise AssertionError("oversized image dimensions were accepted")


def test_production_errors_do_not_expose_internal_details(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", None)
    monkeypatch.setattr(api, "MAX_IMAGE_PIXELS", None)
    monkeypatch.setattr(
        api,
        "enhance_image",
        lambda image: (_ for _ in ()).throw(RuntimeError("private path")),
    )

    try:
        asyncio.run(api.enhance(_upload_image()))
    except HTTPException as error:
        assert error.status_code == 500
        assert error.detail == "Image enhancement failed. Please try again later."
        assert "private path" not in error.detail
    else:
        raise AssertionError("inference failure was not reported")
