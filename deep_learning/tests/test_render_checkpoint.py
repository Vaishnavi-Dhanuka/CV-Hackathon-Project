from hashlib import sha256
from importlib import import_module
from io import BytesIO
import json
from urllib.error import HTTPError

import pytest

checkpoint_setup = import_module("scripts.fetch_render_checkpoint")
TOKEN = "test-secret-token"
ASSET_URL = (
    "https://api.github.com/repos/Vaishnavi-Dhanuka/"
    "CV-Hackathon-Project/releases/assets/123"
)


class _DownloadResponse(BytesIO):
    def geturl(self) -> str:
        return "https://objects.githubusercontent.com/retinexformer.pth"


def _release_response(asset_url: str = ASSET_URL) -> BytesIO:
    return BytesIO(
        json.dumps(
            {
                "assets": [
                    {"name": "unrelated.txt", "url": "https://api.github.com/ignored"},
                    {
                        "name": checkpoint_setup.ASSET_FILENAME,
                        "url": asset_url,
                    },
                ]
            }
        ).encode()
    )


def test_private_release_asset_download_uses_authenticated_api_requests(
    monkeypatch, tmp_path
):
    content = b"verified checkpoint bytes"
    monkeypatch.setattr(
        checkpoint_setup,
        "EXPECTED_SHA256",
        sha256(content).hexdigest(),
    )
    requests = []
    responses = [_release_response(), _DownloadResponse(content)]

    def fake_open(request):
        requests.append(request)
        return responses.pop(0)

    monkeypatch.setattr(checkpoint_setup, "_open", fake_open)
    destination = tmp_path / "model.pth"

    checkpoint_setup.fetch_checkpoint(TOKEN, destination)

    assert destination.read_bytes() == content
    assert list(tmp_path.glob("*.download")) == []
    assert requests[0].full_url == checkpoint_setup.GITHUB_RELEASE_API
    assert requests[0].get_header("Authorization") == f"Bearer {TOKEN}"
    assert requests[0].get_header("Accept") == "application/vnd.github+json"
    assert requests[1].full_url == ASSET_URL
    assert requests[1].get_header("Authorization") == f"Bearer {TOKEN}"
    assert requests[1].get_header("Accept") == "application/octet-stream"


def test_valid_existing_checkpoint_is_not_downloaded(monkeypatch, tmp_path):
    content = b"already verified"
    monkeypatch.setattr(
        checkpoint_setup,
        "EXPECTED_SHA256",
        sha256(content).hexdigest(),
    )
    destination = tmp_path / "model.pth"
    destination.write_bytes(content)
    monkeypatch.setattr(
        checkpoint_setup,
        "_open",
        lambda request: pytest.fail("valid checkpoint should not be fetched again"),
    )

    checkpoint_setup.fetch_checkpoint("", destination)

    assert destination.read_bytes() == content


def test_corrupt_existing_checkpoint_is_replaced_only_after_hash_verification(
    monkeypatch, tmp_path
):
    content = b"verified replacement"
    monkeypatch.setattr(
        checkpoint_setup,
        "EXPECTED_SHA256",
        sha256(content).hexdigest(),
    )
    destination = tmp_path / "model.pth"
    destination.write_bytes(b"corrupt checkpoint")
    responses = [_release_response(), _DownloadResponse(content)]
    monkeypatch.setattr(checkpoint_setup, "_open", lambda request: responses.pop(0))

    checkpoint_setup.fetch_checkpoint(TOKEN, destination)

    assert destination.read_bytes() == content


def test_missing_asset_fails_without_creating_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(
        checkpoint_setup,
        "_open",
        lambda request: BytesIO(b'{"assets":[]}'),
    )
    destination = tmp_path / "model.pth"

    with pytest.raises(RuntimeError, match="does not contain"):
        checkpoint_setup.fetch_checkpoint(TOKEN, destination)

    assert not destination.exists()


def test_authentication_failure_is_actionable_without_echoing_token(
    monkeypatch, tmp_path
):
    def fail_authentication(request):
        raise HTTPError(request.full_url, 401, "Unauthorized", None, None)

    monkeypatch.setattr(checkpoint_setup, "_open", fail_authentication)

    with pytest.raises(RuntimeError, match="invalid, or expired") as error:
        checkpoint_setup.fetch_checkpoint(TOKEN, tmp_path / "model.pth")

    assert TOKEN not in str(error.value)
    assert not (tmp_path / "model.pth").exists()


def test_bad_download_digest_preserves_existing_checkpoint(monkeypatch, tmp_path):
    destination = tmp_path / "model.pth"
    original = b"existing corrupt checkpoint"
    destination.write_bytes(original)
    responses = [_release_response(), _DownloadResponse(b"not the expected model")]
    monkeypatch.setattr(checkpoint_setup, "_open", lambda request: responses.pop(0))

    with pytest.raises(RuntimeError, match="SHA-256"):
        checkpoint_setup.fetch_checkpoint(TOKEN, destination)

    assert destination.read_bytes() == original
    assert list(tmp_path.glob("*.download")) == []
