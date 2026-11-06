
import asyncio
from io import BytesIO
import logging
import os

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from deep_learning import retinex_inference

enhance_image = retinex_inference.enhance_image

logger = logging.getLogger(__name__)
_inference_lock = asyncio.Lock()


def _positive_int_env(name: str) -> int | None:
    value = os.getenv(name)
    if value is None:
        return None
    try:
        parsed = int(value)
    except ValueError as error:
        raise RuntimeError(f"{name} must be a positive integer.") from error
    if parsed <= 0:
        raise RuntimeError(f"{name} must be a positive integer.")
    return parsed


MAX_UPLOAD_BYTES = _positive_int_env("RETINEXFORMER_MAX_UPLOAD_BYTES")
MAX_IMAGE_PIXELS = _positive_int_env("RETINEXFORMER_MAX_IMAGE_PIXELS")


def _is_production() -> bool:
    return os.getenv("APP_ENV", "").lower() == "production"

app = FastAPI(
    title="Low-Light Image Enhancement API",
    version="1.0.0"
)


@app.middleware("http")
async def reject_oversized_enhance_request(request: Request, call_next):
    if (
        MAX_UPLOAD_BYTES is not None
        and request.method == "POST"
        and request.url.path == "/enhance"
    ):
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                request_size = int(content_length)
            except ValueError:
                return JSONResponse(
                    status_code=400,
                    content={"detail": "Invalid Content-Length header."},
                )
            if request_size < 0:
                return JSONResponse(
                    status_code=400,
                    content={"detail": "Invalid Content-Length header."},
                )
            if request_size > MAX_UPLOAD_BYTES + 65536:
                return JSONResponse(
                    status_code=413,
                    content={"detail": "Image upload exceeds the configured size limit."},
                )
    return await call_next(request)


@app.get("/")
def home():
    return {"message": "Low-Light Enhancement API is running"}


@app.get("/health")
def health():
    if (
        not retinex_inference.CONFIG.is_file()
        or not retinex_inference.CHECKPOINT.is_file()
    ):
        raise HTTPException(status_code=503, detail="Model assets are not ready.")
    return {"status": "ok"}


@app.post("/enhance")
async def enhance(file: UploadFile = File(...)):
    try:
        async with _inference_lock:
            if MAX_UPLOAD_BYTES is None:
                contents = await file.read()
            else:
                contents = await file.read(MAX_UPLOAD_BYTES + 1)
                if len(contents) > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail="Image upload exceeds the configured size limit.",
                    )

            try:
                with Image.open(BytesIO(contents)) as uploaded_image:
                    if (
                        MAX_IMAGE_PIXELS is not None
                        and uploaded_image.width * uploaded_image.height
                        > MAX_IMAGE_PIXELS
                    ):
                        raise HTTPException(
                            status_code=413,
                            detail="Image dimensions exceed the configured pixel limit.",
                        )
                    image = uploaded_image.convert("RGB")
            except HTTPException:
                raise
            except (
                Image.DecompressionBombError,
                UnidentifiedImageError,
                OSError,
            ) as error:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid or unsupported image file.",
                ) from error

            enhanced = await run_in_threadpool(enhance_image, image)

            output = BytesIO()
            enhanced.save(output, format="PNG")
            output.seek(0)

            return StreamingResponse(
                output,
                media_type="image/png",
                headers={"Content-Disposition": "attachment; filename=enhanced.png"},
            )
    except HTTPException:
        raise
    except FileNotFoundError as error:
        logger.exception("RetinexFormer model assets are unavailable.")
        detail = (
            "The enhancement model is not configured. Please contact the service "
            "administrator."
            if _is_production()
            else str(error)
        )
        raise HTTPException(status_code=503, detail=detail) from error
    except Exception as error:
        logger.exception("Image enhancement failed.")
        detail = (
            "Image enhancement failed. Please try again later."
            if _is_production()
            else str(error)
        )
        raise HTTPException(status_code=500, detail=detail) from error
    finally:
        await file.close()
