# VisionForge â€” Robust Low-Light Image Restoration and Enhancement

**Problem ID:** P1  
**Team ID:** T21  
**Team name:** VisionForge  
**Repository:** https://github.com/Vaishnavi-Dhanuka/CV-Hackathon-Project  
**Live demo (Streamlit frontend):** [https://lumina-streamlit.onrender.com](https://lumina-streamlit.onrender.com)  

**Deployment note:** This URL identifies the frontend only. The Render Free backend exceeded its 512 MB memory limit during inference; hosted model inference is therefore not confirmed as reliable or production-ready.

## Team

| USN | Member |
|---|---|
| 1RUA24CSE0442 | Shreyank Balaji |
| 1RUA24CSE0472 | Sparsha G |
| 1RUA24CSE0493 | Tanisha Shrikant Gavali |
| 1RUA24CSE0512 | Vaishnavi Dhanuka |

## Overview

VisionForge addresses low-light image restoration and enhancement. The system provides a classical computer-vision baseline, an official pretrained RetinexFormer inference path, and an experimental hybrid path that fuses classical and deep-learning outputs. LUMINA provides a Streamlit interface; a FastAPI service exposes RetinexFormer inference.

The production recommendation from the documented M3 experiments is the standalone pretrained RetinexFormer model. The spatial-adaptive hybrid remains available for comparison; the reported validation and eval15 results do **not** show it outperforming RetinexFormer.

## Features

- Original image preview and download.
- Classical CV enhancement through the M1 guided Retinex path.
- RetinexFormer enhancement through a FastAPI `/enhance` endpoint.
- Experimental weighted, adaptive, and spatial-adaptive fusion utilities.
- Visual display of heuristic fusion contribution maps.
- Image diagnostics and reported image-quality metrics.
- Evaluation tooling for paired validation and held-out comparisons.

## Architecture at a glance

```text
                 Streamlit / LUMINA frontend
                           |
             +-------------+-------------+
             |             |             |
          Original      Classical      Deep learning
           image          CV           API client
                           |               |
                     M1 guided        FastAPI /enhance
                      Retinex              |
                           |          RetinexFormer
                           |               |
                           +-------+-------+
                                   |
                         Experimental fusion
                                   |
                          Preview / download
```

The deployed configuration uses separate frontend and backend services. The frontend receives the backend URL through `LUMINA_DL_API_URL`. See `ARCHITECTURE.md`, `DESIGN.md`, and `SETUP.md`.

## Quickstart

Use Python 3.10 for the documented local environment. Python 3.11 has not been verified for this project. Initialize the RetinexFormer submodule:

```powershell
git submodule update --init --recursive
```

Create and activate an environment, install the project dependencies, install the matching OpenCV contrib headless wheel as described in `SETUP.md`, provision the official checkpoint, then start the API and Streamlit frontend in separate terminals.

```powershell
python -m uvicorn deep_learning.api:app --host 127.0.0.1 --port 8000
python -m streamlit run app/main.py
```

The primary API checkpoint is stored locally at:

`results/checkpoints/official_only/retinexformer_official_baseline.pth`

It is not committed to the repository. Its verified SHA-256 is:

`539bd16c4da6179e45616329f249c4672951b1045193428e1d042c50d4b65a0b`

See `SETUP.md` for installation and checkpoint details. A fresh clone without the separately supplied checkpoint cannot run model inference.

## Evaluation

The M3 evaluation used a fixed 97-pair validation split from original LOL (LOL-v1) `our485` and a subsequent one-shot 15-pair eval15 comparison. The eval15 run produced 45 method rows without recorded inference or metric failures.

| Method | PSNR | SSIM | LPIPS | NIQE |
|---|---:|---:|---:|---:|
| Classical CV | 16.3686 | 0.5048 | 0.3590 | 7.9804 |
| Pretrained RetinexFormer | 21.7720 | 0.8224 | 0.1524 | 3.8765 |
| Spatial Hybrid Intelligence | 18.1390 | 0.5885 | 0.2943 | 7.0345 |

These are means from the documented eval15 run only. NIQE is a no-reference statistic, not a ground-truth fidelity measure. Latency is not restoration quality. See `EVALUATION.md` for protocol, validation results, caveats, and reproduction references.

## Repository documentation

- `ARCHITECTURE.md` â€” components and data flow.
- `DESIGN.md` â€” design decisions and trade-offs.
- `SETUP.md` â€” environment, checkpoint, and run instructions.
- `EVALUATION.md` â€” datasets, metrics, benchmark results, and reproducibility.
- `FAILURE_ANALYSIS.md` â€” limitations and failure modes.
- `eval_P1_T21.md` â€” hackathon evaluation submission.

## Team contribution review

Individual contributions are not itemized in this repository because verified per-member responsibility details were not available in the reviewed source material.

## Submission checklist

- [ ] Verify the public URL from a fresh browser/session and test an unseen image.
- [ ] Confirm all commands in `SETUP.md` on a clean environment.
- [ ] Confirm architecture diagram against the final code.
- [ ] Re-run and archive the documented evaluation before claiming reproducibility.

