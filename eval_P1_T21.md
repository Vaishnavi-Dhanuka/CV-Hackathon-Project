1RUA24CSE0442, 1RUA24CSE0472, 1RUA24CSE0493, 1RUA24CSE0512

# P1 - Robust Low-Light Image Restoration and Enhancement

## Team Information

| Field | Details |
|---|---|
| Team ID | T21 |
| Problem ID | P1 |
| Problem Title | Robust Low-Light Image Restoration and Enhancement |
| Team Name | VisionForge |
| Team Members | Shreyank Balaji; Sparsha G; Tanisha Shrikant Gavali; Vaishnavi Dhanuka |
| GitHub Repository | https://github.com/Vaishnavi-Dhanuka/CV-Hackathon-Project |
| Evaluation Source Revision | `53873b5c79f3759449e723483b172c88ce3a9aa4` (documentation branch snapshot) |
| Live Demo | [https://lumina-streamlit.onrender.com](https://lumina-streamlit.onrender.com) Ã¢â‚¬â€ Streamlit frontend; backend inference reliability is not confirmed. |

## 1. Problem Understanding and Solution Strategy

Low-light images can suffer from poor visibility, noise, colour distortion, and loss of detail. The project provides a comparison-oriented enhancement system with a classical computer-vision path, an official pretrained RetinexFormer path, and an experimental hybrid fusion path.

The documented M3 evaluation selected standalone pretrained RetinexFormer as the production recommendation. The current spatial-adaptive hybrid remains experimental because it did not meet the validation selection rule and underperformed RetinexFormer on the reported eval15 comparison.

## 2. Technical Architecture and Implementation

### 2.1 Classical baseline

The classical adapter calls the M1 `prepare_image()` and guided `retinex_enhance()` functions. It returns an RGB `uint8` image, restores original dimensions when required, and reports branch latency.

### 2.2 Deep-learning method

The FastAPI `/enhance` endpoint accepts an image upload and delegates to `deep_learning/retinex_inference.py`. The inference path uses the official RetinexFormer implementation and LOLv2-real configuration, strictly loads the `params` state dictionary, caches the model, pads dimensions as required, and returns an RGB image at the original dimensions.

The production checkpoint is `results/checkpoints/official_only/retinexformer_official_baseline.pth`, separately provisioned and verified by SHA-256 `539bd16c4da6179e45616329f249c4672951b1045193428e1d042c50d4b65a0b`.

### 2.3 Hybrid method

The hybrid branch obtains classical and deep-learning outputs and combines them using spatial-adaptive heuristic weights. The maps describe blend proportions; they are not learned attention or estimates of output correctness. The evaluation results do not support claiming a quality improvement over RetinexFormer.

### 2.4 Frontend and API

The LUMINA Streamlit frontend offers image preview, pipeline selection, comparison, metrics, explainability visualizations, and PNG download. The FastAPI service provides `/`, `/health`, and `/enhance` endpoints. Local and hosted frontend/backend communication is configured through `LUMINA_DL_API_URL`.

## 3. Experimental Benchmarks and Evaluation

### 3.1 M3 seeded validation

The fixed seed-2026 split used 97 paired validation images from original LOL (LOL-v1) `our485` (388 training names). No fusion candidate met the predeclared paired-bootstrap criteria. Pretrained RetinexFormer was selected as the recommended production method.

| Method | PSNR | SSIM | LPIPS | NIQE |
|---|---:|---:|---:|---:|
| Classical CV | 16.9612 | 0.4878 | 0.4394 | 8.2752 |
| Pretrained RetinexFormer | 22.1058 | 0.8451 | 0.1645 | 4.0858 |
| Weighted 50:50 | 19.8884 | 0.6646 | 0.2922 | 6.2229 |
| Adaptive fusion | 18.8833 | 0.5983 | 0.3457 | 7.0038 |
| Spatial-adaptive fusion | 18.4391 | 0.5711 | 0.3678 | 7.3426 |
| DL-dominant 90:10 | 21.8293 | 0.8286 | 0.1743 | 4.0949 |
| DL-dominant 95:5 | 21.9894 | 0.8399 | 0.1671 | 4.0430 |
| DL-dominant 80:20 | 21.4234 | 0.7932 | 0.1983 | 4.5403 |

### 3.2 One-shot held-out eval15

After freezing the validation decision, 15 paired eval15 images were evaluated, producing 45 method rows without recorded inference or metric failures.

| Method | PSNR | SSIM | LPIPS | NIQE | CIEDE2000 |
|---|---:|---:|---:|---:|---:|
| Classical CV | 16.3686 | 0.5048 | 0.3590 | 7.9804 | 16.2439 |
| Pretrained RetinexFormer | 21.7720 | 0.8224 | 0.1524 | 3.8765 | 8.9765 |
| Hybrid Intelligence (spatial-adaptive) | 18.1390 | 0.5885 | 0.2943 | 7.0345 | 13.7454 |

Hybrid was lower than RetinexFormer by 3.6330 dB PSNR and 0.2339 SSIM, and higher by 0.1419 LPIPS, 3.1580 NIQE, and 4.7690 CIEDE2000. It improved PSNR on 3 of 15 images but did not improve mean quality metrics.

NIQE is a no-reference statistic and is not a ground-truth fidelity metric. Latency is not image quality. Other separately reported experiments (69-image architecture comparison, Model A/B validation, and 100-image official test) must remain separate from these M3 results.

### 3.3 Ablation Study

The M3 validation compared fixed 50:50 weighted fusion, global adaptive fusion, spatial-adaptive fusion, and RetinexFormer-dominant 90:10, 95:5, and 80:20 blends. No candidate met the predeclared paired-bootstrap selection rule. The results therefore do not justify selecting fusion over standalone RetinexFormer.

## 4. Failure Modes and Limitations

- Spatial Hybrid underperformed standalone RetinexFormer on the reported validation and eval15 means.
- Heuristic fusion maps represent blend proportions, not learned attention or pixel-level correctness.
- Classical CV depends on the M1 package and OpenCV contrib.
- CPU inference latency is hardware-dependent; local timing does not establish hosted performance.
- The checkpoint and datasets are separately provisioned and are not included in Git.
- Dataset-specific results do not guarantee performance on all unseen scenes or sensors.
- Specific image-level failure examples must be selected from actual saved outputs and attached before final submission.

## 5. Deployment and Quickstart

Initialize the submodule:

```powershell
git submodule update --init --recursive
```

Provision and verify the separately distributed official checkpoint as described in `LUMINA_SETUP_GUIDE.md`.

Start the API:

```powershell
python -m uvicorn deep_learning.api:app --host 127.0.0.1 --port 8000
```

Start the frontend in another terminal:

```powershell
python -m streamlit run app/main.py
```

The supplied public URL is the Streamlit frontend only. The Render Free backend exceeded its 512 MB memory limit during inference, so successful hosted model inference is not established. Do not represent the frontend URL as proof that the full application is operational.

## 6. Validation and Reproducibility

The recorded local test run completed with 129 passed tests, 23 warnings, and 7 subtests, along with AppTest, API, validation, and eval15 checks. These results must be rerun on the final release candidate. Use the fixed split manifest and commands documented in `LUMINA_EVALUATION_REPORT.md`; do not tune on eval15. The frontend demo URL does not verify hosted backend inference, because the Render Free backend exceeded its 512 MB memory limit during inference.

## 7. Team Contributions

Individual member responsibilities are not itemized here because verified per-member contribution details were not available in the reviewed source material.
