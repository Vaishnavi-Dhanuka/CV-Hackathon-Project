# Setup and Run Guide

## 1. Requirements

Use Python 3.10 for the documented local environment. Python 3.11 has not been verified for this project. The project has separate pinned dependency files for Render backend and frontend.

## 2. Windows PowerShell: local environment

From the repository root:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.2.2 torchvision==0.17.2 `
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

The project setup guide notes that `pyiqa` depends on the non-contrib OpenCV wheel while M1's guided Retinex path requires `cv2.ximgproc`. Install the matching contrib wheel last:

```powershell
python -m pip install --no-deps --force-reinstall `
  opencv-contrib-python-headless==4.11.0.86
python -c "import cv2; assert hasattr(cv2, 'ximgproc'); print(cv2.__version__)"
```

If activation is blocked, use `.\.venv\Scripts\python.exe` directly.

**Compatibility note for reviewer:** The existing root requirements and the pinned Render requirements are different dependency sets. Verify this sequence on a clean Python 3.10 environment before treating it as a guaranteed one-command install.

## 3. Initialize RetinexFormer submodule

```powershell
git submodule update --init --recursive
git -C third_party\Retinexformer rev-parse HEAD
```

Expected revision:

`1e9a0efce4b306b6701b824768370ff26066c32a`

The official configuration must exist at:

`third_party/Retinexformer/Options/RetinexFormer_LOL_v2_real.yml`

The configured RetinexFormer dependency is the `third_party/Retinexformer` submodule. No separate `third_party/Retinexformer_M3_backup/` directory is required by the documented setup.

## 4. Provision the official checkpoint

The primary API checkpoint must be installed at:

`results/checkpoints/official_only/retinexformer_official_baseline.pth`

Expected size: `6,478,393` bytes.  
Expected SHA-256:

`539bd16c4da6179e45616329f249c4672951b1045193428e1d042c50d4b65a0b`

Verify:

```powershell
Get-Item results\checkpoints\official_only\retinexformer_official_baseline.pth
Get-FileHash results\checkpoints\official_only\retinexformer_official_baseline.pth -Algorithm SHA256
```

The checkpoint is not stored in Git. Obtain it from the team's authorized checkpoint distribution. Do not substitute Model A/B or older checkpoint files.

For Render, `render.yaml` runs `scripts/fetch_render_checkpoint.py` before starting Uvicorn. The deployment guide describes a GitHub release asset and a `GITHUB_TOKEN` secret. Confirm the release and secret are configured by the deployment owner; a local checkpoint verification does not verify Render's download.

## 5. Start local services

Terminal 1 â€” API:

```powershell
python -m uvicorn deep_learning.api:app --host 127.0.0.1 --port 8000
```

Terminal 2 â€” frontend:

```powershell
python -m streamlit run app/main.py
```

The DL client defaults to `http://127.0.0.1:8000`. To point the frontend to another API:

```powershell
$env:LUMINA_DL_API_URL = "http://127.0.0.1:8000"  # replace with the actual backend URL when using a hosted API
python -m streamlit run app/main.py
```

For local development, keep `http://127.0.0.1:8000`. When using a separately hosted backend, set `LUMINA_DL_API_URL` to that service's actual URL. Do not commit secrets or private service credentials.

## 6. Render deployment

The Blueprint defines separate services `lumina-retinex-api` and `lumina-streamlit`. The frontend obtains the backend URL through `LUMINA_DL_API_URL` using the backend service's `RENDER_EXTERNAL_URL`.

`render.yaml` currently declares both services on the Free plan. The Render Free backend exceeded its 512 MB memory limit during inference. The frontend URL alone does not establish that hosted inference works; CPU inference, cold-start time, memory, and large-image latency remain unverified on the hosted plan.

## 7. Tests and checks

```powershell
python -m pytest evaluation/tests hybrid/tests app/tests -v
git diff --check
```

The recorded local test run completed with 129 passed, 23 warnings, and 7 subtests passed. Treat that as a recorded run; rerun the command on the final commit candidate and retain the terminal output.

Mocked model-loading tests do not replace a checkpoint-backed inference smoke test. The checkpoint-backed test must be run separately.

## 8. Evaluation reproduction

Use the commands and manifest documented in `evaluation/README.md`. The seeded `our485` validation and one-shot eval15 finalizer are separate steps. Do not tune using eval15. The finalizer refuses existing output paths; use unique output paths for new runs.

See `EVALUATION.md` for metric definitions, reported results, and caveats.

