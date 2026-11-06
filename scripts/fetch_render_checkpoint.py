#!/usr/bin/env python3
"""Fetch and verify the official RetinexFormer checkpoint for Render startup."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import (
    HTTPRedirectHandler,
    Request,
    build_opener,
)

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = (
    ROOT
    / "results"
    / "checkpoints"
    / "official_only"
    / "retinexformer_official_baseline.pth"
)
GITHUB_RELEASE_API = (
    "https://api.github.com/repos/Vaishnavi-Dhanuka/"
    "CV-Hackathon-Project/releases/tags/model-checkpoint-v1"
)
ASSET_FILENAME = "retinexformer_official_baseline.pth"
EXPECTED_SHA256 = "539bd16c4da6179e45616329f249c4672951b1045193428e1d042c50d4b65a0b"
MAX_DOWNLOAD_BYTES = 8 * 1024 * 1024
GITHUB_API_HOST = "api.github.com"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _SafeRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, url):
        if urlparse(url).scheme != "https":
            raise HTTPError(url, code, "Refusing non-HTTPS redirect.", headers, response)

        redirected = super().redirect_request(
            request, response, code, message, headers, url
        )
        if redirected is not None and (
            urlparse(request.full_url).hostname != urlparse(url).hostname
        ):
            redirected.remove_header("Authorization")
        return redirected


def _open(request: Request):
    return build_opener(_SafeRedirectHandler()).open(request, timeout=60)


def _raise_http_error(stage: str, status: int) -> RuntimeError:
    if status == 401:
        reason = "GITHUB_TOKEN is missing, invalid, or expired."
    elif status == 403:
        reason = "GITHUB_TOKEN lacks release access, or GitHub rate-limited the request."
    elif status == 404:
        reason = "The tagged release or its asset was not found or is not accessible."
    else:
        reason = f"GitHub returned HTTP {status}."
    return RuntimeError(f"GitHub {stage} failed. {reason}")


def _get_asset_url(token: str) -> str:
    request = Request(
        GITHUB_RELEASE_API,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "User-Agent": "lumina-render-deployment",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with _open(request) as response:
            release = json.load(response)
    except HTTPError as error:
        raise _raise_http_error("release lookup", error.code) from None
    except (URLError, TimeoutError, OSError, json.JSONDecodeError):
        raise RuntimeError(
            "GitHub release lookup failed. Check network access and try again."
        ) from None

    assets = release.get("assets") if isinstance(release, dict) else None
    if not isinstance(assets, list):
        raise RuntimeError("GitHub release lookup returned an invalid response.")

    for asset in assets:
        if isinstance(asset, dict) and asset.get("name") == ASSET_FILENAME:
            asset_url = asset.get("url")
            parsed_url = urlparse(asset_url) if isinstance(asset_url, str) else None
            if (
                parsed_url is None
                or parsed_url.scheme != "https"
                or parsed_url.hostname != GITHUB_API_HOST
            ):
                raise RuntimeError(
                    "GitHub release returned an invalid checkpoint asset URL."
                )
            return asset_url

    raise RuntimeError(
        f"GitHub release model-checkpoint-v1 does not contain {ASSET_FILENAME}."
    )


def fetch_checkpoint(token: str, destination: Path = CHECKPOINT) -> None:
    if destination.is_file() and _sha256(destination) == EXPECTED_SHA256:
        print("Verified RetinexFormer checkpoint already exists.")
        return
    if not token:
        raise RuntimeError("GITHUB_TOKEN is required to download the private release.")

    asset_url = _get_asset_url(token)
    request = Request(
        asset_url,
        headers={
            "Accept": "application/octet-stream",
            "Authorization": f"Bearer {token}",
            "User-Agent": "lumina-render-deployment",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    digest = hashlib.sha256()
    total_bytes = 0

    try:
        try:
            with _open(request) as response:
                if urlparse(response.geturl()).scheme != "https":
                    raise RuntimeError("Checkpoint download redirected away from HTTPS.")
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    prefix=".retinexformer-",
                    suffix=".download",
                    dir=destination.parent,
                    delete=False,
                ) as temporary_file:
                    temporary_path = Path(temporary_file.name)
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        total_bytes += len(chunk)
                        if total_bytes > MAX_DOWNLOAD_BYTES:
                            raise RuntimeError(
                                "Checkpoint download exceeded the size limit."
                            )
                        digest.update(chunk)
                        temporary_file.write(chunk)
        except HTTPError as error:
            raise _raise_http_error("asset download", error.code) from None
        except (URLError, TimeoutError, OSError):
            raise RuntimeError(
                "GitHub checkpoint download failed. Check network access and try again."
            ) from None

        if digest.hexdigest() != EXPECTED_SHA256:
            raise RuntimeError(
                "Downloaded checkpoint SHA-256 did not match the verified model."
            )

        if destination.is_file() and _sha256(destination) == EXPECTED_SHA256:
            print("Verified RetinexFormer checkpoint already exists.")
            return
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    print(f"Downloaded and verified RetinexFormer checkpoint ({total_bytes} bytes).")


if __name__ == "__main__":
    if CHECKPOINT.is_file() and _sha256(CHECKPOINT) == EXPECTED_SHA256:
        print("Verified RetinexFormer checkpoint already exists.")
        raise SystemExit(0)

    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        raise SystemExit(
            "GITHUB_TOKEN is required to download the private checkpoint release."
        )
    try:
        fetch_checkpoint(github_token)
    except RuntimeError as error:
        raise SystemExit(f"Checkpoint setup failed: {error}") from None
