# M3 Setup and Run Guide

This guide covers the LUMINA frontend, M3 adapters/evaluation, and the official
RetinexFormer API integration. The production checkpoint is distributed
separately and is intentionally not stored in Git.

## Windows PowerShell setup

Run from the repository root on Python 3.11:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.2.2 torchvision==0.17.2 `
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

`pyiqa` depends on the non-contrib OpenCV wheel, while M1's existing guided
Retinex path requires `cv2.ximgproc`. Install the matching contrib wheel last
so the contrib bindings are the active `cv2` module:

```powershell
python -m pip install --no-deps --force-reinstall `
  opencv-contrib-python-headless==4.11.0.86
python -c "import cv2; assert hasattr(cv2, 'ximgproc'); print(cv2.__version__)"
```

If PowerShell blocks venv activation, enable it for the current process or
invoke `.\.venv\Scripts\python.exe` directly.

## RetinexFormer submodule

Initialize the registered submodule and verify its pinned revision:

```powershell
git submodule update --init --recursive
git -C third_party\Retinexformer rev-parse HEAD
```

Expected submodule commit:
`1e9a0efce4b306b6701b824768370ff26066c32a`.

The official configuration must exist at
`third_party/Retinexformer/Options/RetinexFormer_LOL_v2_real.yml`.
Do not delete `third_party/Retinexformer_M3_backup/` as part of setup; it is a
local backup, not the configured submodule.

## Official checkpoint

The separately supplied checkpoint must be installed manually at:

```text
results/checkpoints/official_only/retinexformer_official_baseline.pth
```

Obtain it only from the team's authorized checkpoint distribution; this
repository does not contain a download URL or an alternate model/checkpoint.
Never replace it with an experimental Model A/B checkpoint or an older
RetinexFormer weight file.

Verify presence and SHA-256:

```powershell
Get-Item results\checkpoints\official_only\retinexformer_official_baseline.pth
Get-FileHash results\checkpoints\official_only\retinexformer_official_baseline.pth -Algorithm SHA256
```

The verified project checkpoint is 6,478,393 bytes with SHA-256
`539bd16c4da6179e45616329f249c4672951b1045193428e1d042c50d4b65a0b`.
Inference loads this checkpoint once, strictly from its `params` state
dictionary, and uses CPU when CUDA is unavailable. A fresh clone without the
separately supplied checkpoint cannot run model inference.

## Start the services

Start the API in one PowerShell window:

```powershell
python -m uvicorn deep_learning.api:app --host 127.0.0.1 --port 8000
```

Start LUMINA in a second window:

```powershell
python -m streamlit run app/main.py
```

The frontend exposes Original Image, Classical CV, Official RetinexFormer, and
Hybrid Intelligence modes. The Hybrid path invokes both M3 adapters, then the
current spatial-adaptive fusion, and shows its actual returned contribution
maps. It does not interpret contribution as restoration quality.

## Tests

Run the M3 test suites:

```powershell
python -m pytest evaluation/tests hybrid/tests app/tests -v
git diff --check
```

Model loading tests mock model construction and do not download checkpoints.
The checkpoint-backed CPU smoke test is separate from the mocked tests.

## Validation and held-out evaluation

Use the seeded our485 split to compare fusion candidates, without tuning on
eval15. See [evaluation/README.md](./evaluation/README.md) for the manifest,
validation, and one-shot eval15 commands and output paths.

The eval15 finalizer requires new output paths and preserves
`evaluation/results/lolv2_eval15_official_checkpoint.csv`.
