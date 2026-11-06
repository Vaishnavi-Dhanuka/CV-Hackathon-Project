import asyncio
import io
import re
import sys
import types
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from PIL import Image
from starlette.datastructures import UploadFile

from deep_learning import retinex_inference as inference


@pytest.fixture(autouse=True)
def clear_model_cache():
    inference._load_model.cache_clear()
    yield
    inference._load_model.cache_clear()


def install_basicsr_mocks(monkeypatch, create_model=None, parse=None):
    basicsr = types.ModuleType("basicsr")
    models = types.ModuleType("basicsr.models")
    utils = types.ModuleType("basicsr.utils")
    options = types.ModuleType("basicsr.utils.options")
    models.create_model = create_model or Mock()
    options.parse = parse or Mock(return_value={})

    monkeypatch.setitem(sys.modules, "basicsr", basicsr)
    monkeypatch.setitem(sys.modules, "basicsr.models", models)
    monkeypatch.setitem(sys.modules, "basicsr.utils", utils)
    monkeypatch.setitem(sys.modules, "basicsr.utils.options", options)
    return models.create_model, options.parse


def configure_fake_model(monkeypatch, tmp_path):
    checkpoint_path = tmp_path / "retinexformer_official_baseline.pth"
    checkpoint_path.touch()
    monkeypatch.setattr(inference, "CHECKPOINT", checkpoint_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    network = torch.nn.Identity()
    load_state_dict = Mock(wraps=network.load_state_dict)
    monkeypatch.setattr(network, "load_state_dict", load_state_dict)
    create_model = Mock(return_value=types.SimpleNamespace(net_g=network))
    parse = Mock(return_value={})
    install_basicsr_mocks(monkeypatch, create_model, parse)
    torch_load = Mock(return_value={"params": network.state_dict()})
    monkeypatch.setattr(torch, "load", torch_load)
    return network, load_state_dict, create_model, parse, torch_load


def test_official_checkpoint_path_and_params_format():
    expected_root = inference.Path(__file__).resolve().parents[2]
    assert inference.CHECKPOINT == (
        expected_root
        / "results"
        / "checkpoints"
        / "official_only"
        / "retinexformer_official_baseline.pth"
    )
    checkpoint = torch.load(
        inference.CHECKPOINT, map_location="cpu", weights_only=True
    )
    assert "params" in checkpoint
    assert isinstance(checkpoint["params"], dict)


def test_single_image_inference_returns_rgb_uint8_original_dimensions(
    monkeypatch, tmp_path
):
    _, load_state_dict, create_model, parse, torch_load = configure_fake_model(
        monkeypatch, tmp_path
    )
    image = Image.new("RGBA", (7, 5), (40, 120, 220, 0))

    result = inference.enhance_image(image)

    assert result.mode == "RGB"
    assert result.size == image.size
    output = np.asarray(result)
    assert output.dtype == np.uint8
    assert output.shape == (5, 7, 3)
    np.testing.assert_array_equal(output, np.full((5, 7, 3), (40, 120, 220), dtype=np.uint8))
    parse.assert_called_once_with(str(inference.CONFIG), is_train=False)
    assert create_model.call_args.args[0]["dist"] is False
    assert create_model.call_args.args[0]["num_gpu"] == 0
    torch_load.assert_called_once_with(
        inference.CHECKPOINT, map_location="cpu", weights_only=False
    )
    load_state_dict.assert_called_once_with(
        torch_load.return_value["params"], strict=True
    )


def test_model_is_cached_and_uses_cpu(monkeypatch, tmp_path):
    _, _, create_model, _, torch_load = configure_fake_model(monkeypatch, tmp_path)

    first = inference.enhance_image(Image.new("RGB", (4, 4), "red"))
    second = inference.enhance_image(Image.new("RGB", (4, 4), "blue"))

    assert first.size == second.size == (4, 4)
    create_model.assert_called_once()
    torch_load.assert_called_once()
    assert inference._load_model()[1].type == "cpu"


def test_missing_official_checkpoint_raises_without_fallback(
    monkeypatch, tmp_path
):
    missing_checkpoint = tmp_path / "missing-official-checkpoint.pth"
    monkeypatch.setattr(inference, "CHECKPOINT", missing_checkpoint)
    create_model, _ = install_basicsr_mocks(monkeypatch)
    torch_load = Mock()
    monkeypatch.setattr(torch, "load", torch_load)

    with pytest.raises(FileNotFoundError, match=re.escape(str(missing_checkpoint))):
        inference._load_model()

    create_model.assert_not_called()
    torch_load.assert_not_called()


def test_model_loading_error_propagates_without_fallback(monkeypatch, tmp_path):
    checkpoint_path = tmp_path / "retinexformer_official_baseline.pth"
    checkpoint_path.touch()
    monkeypatch.setattr(inference, "CHECKPOINT", checkpoint_path)
    create_model = Mock(side_effect=RuntimeError("model initialization failed"))
    install_basicsr_mocks(monkeypatch, create_model=create_model)
    monkeypatch.setattr(torch, "load", Mock())

    with pytest.raises(RuntimeError, match="model initialization failed"):
        inference._load_model()

    create_model.assert_called_once()


def test_invalid_non_pil_input_raises_type_error():
    with pytest.raises(TypeError, match="PIL.Image.Image"):
        inference.enhance_image(np.zeros((2, 2, 3), dtype=np.uint8))


def test_api_returns_png_compatible_with_existing_client(monkeypatch):
    from deep_learning import api

    source = Image.new("RGB", (3, 2), (10, 20, 30))
    input_buffer = io.BytesIO()
    source.save(input_buffer, format="PNG")
    monkeypatch.setattr(
        api,
        "enhance_image",
        lambda image: Image.new("RGB", image.size, (200, 150, 100)),
    )
    upload = UploadFile(
        filename="input.png", file=io.BytesIO(input_buffer.getvalue())
    )

    async def read_response():
        response = await api.enhance(upload)
        body = b"".join([chunk async for chunk in response.body_iterator])
        return response, body

    response, body = asyncio.run(read_response())
    assert response.media_type == "image/png"
    with Image.open(io.BytesIO(body)) as output:
        assert output.format == "PNG"
        assert output.mode == "RGB"
        assert output.size == source.size
        assert output.getpixel((0, 0)) == (200, 150, 100)


def test_api_rejects_invalid_image_bytes():
    from deep_learning import api
    from fastapi import HTTPException

    upload = UploadFile(filename="invalid.png", file=io.BytesIO(b"not an image"))

    with pytest.raises(HTTPException) as error:
        asyncio.run(api.enhance(upload))

    assert error.value.status_code == 400


def test_api_reports_inference_failure(monkeypatch):
    from deep_learning import api
    from fastapi import HTTPException

    source = Image.new("RGB", (2, 2), "black")
    input_buffer = io.BytesIO()
    source.save(input_buffer, format="PNG")
    monkeypatch.setattr(
        api,
        "enhance_image",
        Mock(side_effect=RuntimeError("model inference failed")),
    )
    upload = UploadFile(
        filename="input.png", file=io.BytesIO(input_buffer.getvalue())
    )

    with pytest.raises(HTTPException) as error:
        asyncio.run(api.enhance(upload))

    assert error.value.status_code == 500
    assert error.value.detail == "model inference failed"
