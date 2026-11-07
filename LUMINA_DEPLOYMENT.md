# Render deployment

The root `render.yaml` defines two independent Python web services:

- `lumina-retinex-api`: FastAPI with the official pretrained RetinexFormer.
- `lumina-streamlit`: the Streamlit application at `app/main.py`.

The Blueprint uses separate, pinned runtime requirements and Python 3.11.
The backend installs the checked-out RetinexFormer submodule directly; it does
not install the training environment or build CUDA/C++ extensions.

## Before deploying

1. Push the repository, including its `.gitmodules` file and submodule gitlink,
   to a Git provider Render can access. The Render build command explicitly runs
   `git submodule update --init --recursive`; the registered submodule URL is
   public. Confirm the configured revision with:

   ```sh
   git -C third_party/Retinexformer rev-parse HEAD
   ```

   The expected revision is
   `1e9a0efce4b306b6701b824768370ff26066c32a`.

2. Host the authorized checkpoint at a stable HTTPS download endpoint. In the
   `lumina-retinex-api` Render service, set the secret environment variable
   `GITHUB_TOKEN` to a fine-grained token with read-only Contents access to the
   private repository. Store it as a Render secret; never add it to this
   repository or build command. The service uses it to query the
   `model-checkpoint-v1` release and download the named asset.

3. The startup downloader writes only to
   `results/checkpoints/official_only/retinexformer_official_baseline.pth`,
   verifies SHA-256
   `539bd16c4da6179e45616329f249c4672951b1045193428e1d042c50d4b65a0b`,
   and refuses to overwrite a different existing file. Download failures stop
   startup rather than allowing a model fallback.

## Render service settings

The Blueprint configures these commands and settings:

| Service | Build command | Start command | Health check |
|---|---|---|---|
| Backend | `git submodule update --init --recursive && python -m pip install -r requirements-render.txt` | `python scripts/fetch_render_checkpoint.py && uvicorn deep_learning.api:app --host 0.0.0.0 --port $PORT` | `/health` |
| Frontend | `python -m pip install -r requirements-render-frontend.txt` | `python -m streamlit run app/main.py --server.address 0.0.0.0 --server.port $PORT --server.headless true --server.maxUploadSize 10` | `/` |

Both services currently request Render's `free` plan in `render.yaml`. The local
checkpoint-backed CPU smoke test reached 522.9 MiB RSS, so the backend may
exceed the available memory on a Free instance. Verify the current memory
allocation and confirm the required plan with the deployment owner before
relying on the hosted service. A 2-GiB-or-larger backend is recommended.
The frontend is a separate process and does not load the PyTorch model.

## Environment variables

| Service | Variable | Value |
|---|---|---|
| Backend | `APP_ENV` | `production` (set by the Blueprint; suppresses internal exception details in HTTP responses) |
| Backend | `GITHUB_TOKEN` | Required secret with read-only Contents access to the private repository hosting the release |
| Backend | `RETINEXFORMER_MAX_UPLOAD_BYTES` | `10485760` (10 MiB) |
| Backend | `RETINEXFORMER_MAX_IMAGE_PIXELS` | `4194304` (4 megapixels) |
| Frontend | `LUMINA_DL_API_URL` | Automatically set from the backend service URL; no localhost fallback is used in the deployed service |

When running locally, the two image limits are unset unless explicitly
configured, preserving the existing local upload behavior. The API serializes
inference to avoid simultaneous model activation copies; it runs inference in a
worker thread so `/health` remains responsive.

The API rejects oversized requests with a `Content-Length` header before
multipart parsing and independently caps the uploaded file and decoded pixel
count. Starlette may spool multipart data to temporary files; the handler closes
each uploaded file in `finally`. A chunked request without `Content-Length` is
limited after multipart parsing, so monitor temporary-disk use if exposing the
API to untrusted direct clients. The Streamlit uploader also has a 10-MiB limit.
The existing frontend HTTP client timeout is 120 seconds. No Render-side
inference timeout or latency benchmark was verified; large/slow CPU inferences
may exceed a caller or platform request timeout.

## Verification checklist

- [ ] Render completes both service builds, including submodule initialization.
- [ ] Backend logs report a verified checkpoint; its digest matches the value
  above. Do not copy the checkpoint into Git.
- [ ] `GET https://<backend-host>/` returns the API running message.
- [ ] `GET https://<backend-host>/health` returns `{"status":"ok"}`.
- [ ] `GET https://<backend-host>/docs` loads the FastAPI OpenAPI page.
- [ ] A valid sample image sent to `POST /enhance` returns a PNG with matching
  dimensions; invalid files return 400 and over-limit files/images return 413.
- [ ] A failing model request in production returns a generic error, not a local
  filesystem path or exception text.
- [ ] The Streamlit service loads and its official RetinexFormer and Hybrid modes
  call the deployed backend URL through `LUMINA_DL_API_URL`.
- [ ] Measure inference latency and peak memory using representative maximum
  size images on the selected Render plan before opening the service publicly.

## Known limitations

The CPU smoke test was run locally on Windows, not on Render. It covered a
256x256 image only; Render CPU latency, peak memory for larger images, build
duration, and cold-start behavior remain unverified. Model construction occurs
on the first inference request, so that request may be slower than subsequent
ones. The 10-MiB/4-MP limits are conservative deployment safeguards, not
validated quality or capacity thresholds. Streamlit calls the API server-side,
so browser CORS configuration is not required.

## Render Free Backend Memory Limitation

The Render Free backend exceeded its 512 MB memory limit during model
inference. Therefore, the Free backend deployment must not be considered
stable or production-ready. Inference requests may fail when the service
runs out of memory. A hosting plan or environment with sufficient memory
is required for reliable model inference.

The public Streamlit frontend is available at:
https://lumina-streamlit.onrender.com

The frontend being accessible does not guarantee that backend inference
is operational.