1RUA24CSE0442, 1RUA24CSE0472, 1RUA24CSE0493, 1RUA24CSE0512

# P1 - Robust Low-Light Image Restoration and Enhancement

## Team Information

| Field             | Details                                                                                                                                |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| Team ID           | T21                                                                                                                                    |
| Problem ID        | P1                                                                                                                                     |
| Problem Title     | Robust Low-Light Image Restoration and Enhancement                                                                                     |
| Team Name         | VisionForge                                                                                                                            |
| Team Members      | Shreyank Balaji (1RUA24CSE0442); Sparsha G (1RUA24CSE0472); Tanisha Shrikant Gavali (1RUA24CSE0493); Vaishnavi Dhanuka (1RUA24CSE0512) |
| GitHub Repository | https://github.com/Vaishnavi-Dhanuka/CV-Hackathon-Project                                                                              |
| Live Demo         | https://lumina-streamlit.onrender.com — Streamlit frontend; hosted backend inference is not confirmed.                                 |
| Commit evaluated  | `34edaae950027f6d8308a8d05bcc7f1cf548a6d8` (commit recorded in the downloaded archive of `main`; update if later commits are made)     |

## 1. Problem Understanding and Solution Strategy

Low-light images can suffer from poor visibility, noise, colour distortion, and loss of detail. The project provides a comparison-oriented enhancement system with a classical computer-vision path, an official pretrained RetinexFormer path, and an experimental hybrid fusion path.

The documented M3 evaluation selected standalone pretrained RetinexFormer as the production recommendation. The current spatial-adaptive hybrid remains experimental because it did not meet the validation selection rule and underperformed RetinexFormer on the reported eval15 comparison.

## 2. Technical Architecture and Implementation

### 2.1 Classical baseline

The classical adapter calls the M1 `prepare_image()` and guided `retinex_enhance()` functions with default parameters. It returns an RGB `uint8` image, restores original dimensions when required, and reports branch latency. This deployed branch is Retinex only: it does not run denoising, CLAHE, colour correction or the adaptive parameter rule. The fuller classical pipeline (`classical/classical_pipeline.py`) exists in the repository and is evaluated separately in §4, but the app, the runner and the hybrid do not call it.

### 2.2 Deep-learning method

The FastAPI `/enhance` endpoint accepts an image upload and delegates to `deep_learning/retinex_inference.py`. The inference path uses the official RetinexFormer implementation and LOLv2-real configuration, strictly loads the `params` state dictionary, caches the model, pads dimensions as required, and returns an RGB image at the original dimensions.

The production checkpoint is `results/checkpoints/official_only/retinexformer_official_baseline.pth`, separately provisioned and verified by SHA-256 `539bd16c4da6179e45616329f249c4672951b1045193428e1d042c50d4b65a0b`.

### 2.3 Hybrid method

The hybrid branch obtains classical and deep-learning outputs and combines them using spatial-adaptive heuristic weights. The maps describe blend proportions; they are not learned attention or estimates of output correctness. The weights are clipped to [0.25, 0.75], so the classical branch always contributes at least 25% and the blend cannot fall back to RetinexFormer alone. The evaluation results do not support claiming a quality improvement over RetinexFormer.

### 2.4 Frontend and API

The LUMINA Streamlit frontend offers image preview, four pipeline modes (Original Image, Classical CV, Official RetinexFormer, Hybrid Intelligence), comparison, metrics, fusion-contribution maps, and PNG download. Its "Intermediate results" panels (illumination map, Retinex, CLAHE, denoised) are placeholders that report "Not available yet". The FastAPI service provides `/`, `/health`, and `/enhance` endpoints; upload-size and pixel limits apply only when the environment variables are set (the Render blueprint sets them). Local and hosted frontend/backend communication is configured through `LUMINA_DL_API_URL`.

## 3. Experimental Benchmarks and Evaluation

### 3.1 M3 seeded validation

The fixed seed-2026 split used 97 paired validation images from the `our485` folder (`data/LOLdataset/our485`; the repo's manifest labels it "LOLv2 our485", while the folder is the original LOL set) (388 training names). The pretrained checkpoint is a LOL-v2-Real model, so this is a cross-dataset evaluation for it; see `M3_EVALUATION_REPORT.md` and `evaluation/README.md`. No fusion candidate met the predeclared paired-bootstrap criteria. Pretrained RetinexFormer was selected as the recommended production method.

| Method                      |    PSNR |   SSIM |  LPIPS |   NIQE | Inference ms | Fusion ms |
| --------------------------- | ------: | -----: | -----: | -----: | -----------: | --------: |
| Classical CV (Retinex only) | 16.9612 | 0.4878 | 0.4394 | 8.2752 |        77.35 |       N/A |
| Pretrained RetinexFormer    | 22.1058 | 0.8451 | 0.1645 | 4.0858 |      2057.05 |       N/A |
| Weighted 50:50              | 19.8884 | 0.6646 | 0.2922 | 6.2229 |      2134.40 |     11.73 |
| Adaptive fusion             | 18.8833 | 0.5983 | 0.3457 | 7.0038 |      2134.40 |     18.15 |
| Spatial-adaptive fusion     | 18.4391 | 0.5711 | 0.3678 | 7.3426 |      2134.40 |     49.93 |
| DL-dominant 90:10           | 21.8293 | 0.8286 | 0.1743 | 4.0949 |      2134.40 |     12.51 |
| DL-dominant 95:5            | 21.9894 | 0.8399 | 0.1671 | 4.0430 |      2134.40 |     12.57 |
| DL-dominant 80:20           | 21.4234 | 0.7932 | 0.1983 | 4.5403 |      2134.40 |     13.04 |

### 3.2 eval15 comparison (held out from fusion selection)

After freezing the validation decision, the 15 paired LOL (v1) `eval15` images were evaluated, producing 45 method rows without recorded inference or metric failures.

| Method                                 |    PSNR |   SSIM |  LPIPS |   NIQE | Latency ms (inference) | Fusion ms | Noise suppression | Edge preservation | CIEDE2000 (extra) |
| -------------------------------------- | ------: | -----: | -----: | -----: | ---------------------: | --------: | ----------------- | ----------------- | ----------------: |
| Classical CV (Retinex only)            | 16.3686 | 0.5048 | 0.3590 | 7.9804 |                  88.37 |       N/A | not measured      | not measured      |           16.2439 |
| Pretrained RetinexFormer               | 21.7720 | 0.8224 | 0.1524 | 3.8765 |                1960.97 |       N/A | not measured      | not measured      |            8.9765 |
| Hybrid Intelligence (spatial-adaptive) | 18.1390 | 0.5885 | 0.2943 | 7.0345 |                2395.17 |     47.51 | not measured      | not measured      |           13.7454 |

Source: `evaluation/results/lolv2_eval15_m3_final_seed2026.csv` (per image) and `.../charts/summary.txt`. Columns follow the order in the hackathon instruction sheet. Noise and edge values for the full classical pipeline are in §4.2.

Hybrid was lower than RetinexFormer by 3.6330 dB PSNR and 0.2339 SSIM, and higher by 0.1419 LPIPS, 3.1580 NIQE, and 4.7690 CIEDE2000. It improved PSNR on 2 of 15 images (547.png, 179.png), SSIM on 0 and LPIPS on 1 (179.png), recomputed from the per-image CSV; `M3_EVALUATION_REPORT.md` says 3 of 15, which the CSV does not support and should be corrected, and it did not improve mean quality metrics.

NIQE is a no-reference statistic and is not a ground-truth fidelity metric. Latency is not image quality. Other separately reported experiments (69-image architecture comparison, Model A/B validation, and 100-image official test) must remain separate from these M3 results. The 69-image set is held out from the LOL-v2 Real Train split that the pretrained weights were trained on, so it is not an independent test of the baseline (`README.md` §4; not an independent test of the baseline).

## 4. Ablation Study

### 4.1 Fusion ablation (M3 validation, 97 LOL v1 images)

The M3 validation compared fixed 50:50 weighted fusion, global adaptive fusion, spatial-adaptive fusion, and RetinexFormer-dominant 90:10, 95:5, and 80:20 blends (table in §3.1). No candidate met the predeclared paired-bootstrap selection rule. The closest, 95:5, scored 21.99 dB against 22.11 dB for RetinexFormer alone. The results therefore do not justify selecting fusion over standalone RetinexFormer.

### 4.2 Classical pipeline component ablation (separate harness)

Each row removes one stage of the full classical pipeline (`classical/classical_pipeline.py`), which is not the "Classical CV" row in §3. Source: `results/eval_eval15.md` and `results/eval_lolv2_real.md` (per-image: `results/eval_*_per_image.csv`). dPSNR is relative to FULL; positive means removing the component raised PSNR. Noise is the Immerkaer estimate of the output; edge corr is the Pearson correlation of Sobel gradients with the ground truth.

| Config                | eval15 (15 images) PSNR / SSIM / noise / edge corr |  dPSNR | LOL-v2 Real Test (100 images) PSNR / SSIM / noise / edge corr |  dPSNR |
| --------------------- | -------------------------------------------------- | -----: | ------------------------------------------------------------- | -----: |
| Input (no processing) | 7.77 / 0.191 / 1.22 / 0.833                        | -10.20 | 9.72 / 0.196 / 1.52 / 0.818                                   | -10.75 |
| FULL pipeline         | 17.97 / 0.765 / 2.38 / 0.888                       |   0.00 | 20.47 / 0.784 / 3.06 / 0.875                                  |   0.00 |
| - adaptive Retinex    | 19.03 / 0.756 / 2.84 / 0.869                       |  +1.05 | 17.52 / 0.744 / 3.70 / 0.841                                  |  -2.95 |
| - colour correction   | 17.02 / 0.742 / 2.31 / 0.892                       |  -0.96 | 20.71 / 0.771 / 2.98 / 0.886                                  |  +0.24 |
| - CLAHE               | 17.97 / 0.766 / 2.32 / 0.889                       |  -0.01 | 20.62 / 0.783 / 2.96 / 0.875                                  |  +0.14 |
| - denoise             | 17.13 / 0.519 / 8.53 / 0.874                       |  -0.84 | 18.44 / 0.514 / 11.09 / 0.839                                 |  -2.03 |
| - Retinex             | 9.09 / 0.326 / 0.32 / 0.641                        |  -8.88 | 11.34 / 0.338 / 0.44 / 0.628                                  |  -9.14 |
| Retinex only          | 17.16 / 0.477 / 9.32 / 0.867                       |  -0.81 | 16.72 / 0.459 / 11.90 / 0.837                                 |  -3.75 |

Findings:

- Retinex illumination correction is the dominant component (about 9 dB on both sets).
- Denoising protects SSIM (0.765 to 0.519 and 0.784 to 0.514 when removed) and keeps the noise estimate low.
- CLAHE contributes almost nothing (-0.01 and +0.14 dB when removed).
- Colour correction and the adaptive rule give opposite signs on the two sets, and eval15 has only 15 images, so neither should be presented as an established gain.
- A latency cost is attached to each stage: FULL averages 199 ms (eval15) and 169 ms (100 images) in this harness.

## 5. Failure Modes and Limitations

Documented cases (evidence: `results/failure_cases.png`, `results/wild_sheets/`, `evaluation/results/lolv2_eval15_m3_final_seed2026.csv`; no separate failure-analysis file exists in the repo yet):

- **F1, full classical pipeline:** the three lowest of 100 LOL-v2 Real Test images (13.64 to 13.75 dB) come out brighter and lower in contrast than the ground truth, with washed-out warm highlights and a smoothed floor (`results/failure_cases.png`).
- **F2, hybrid:** on eval15 55.png, Hybrid scores 13.94 dB against 21.87 dB for RetinexFormer; classical and hybrid show a green-yellow cast.
- **F3, RetinexFormer:** on eval15 547.png, RetinexFormer (19.46 dB) is desaturated and trails classical (23.50 dB) and hybrid (24.72 dB).
- **F4, RetinexFormer:** on eval15 111.png, RetinexFormer's lowest score (17.84 dB); all methods stay darker than the ground truth.
- **F5, unseen photos (full classical pipeline):** blotchy, posterized backgrounds on near-black scenes, blown-out lamps with bluish haze, and unsuppressed noise (`results/wild_sheets/wild_comparison_1.png`). RetinexFormer and Hybrid have not yet been run on these photos.

General limitations:

- Spatial Hybrid underperformed standalone RetinexFormer on the reported validation and eval15 means; the fusion weights are clipped to [0.25, 0.75], so the classical branch always contributes at least 25%.
- Heuristic fusion maps represent blend proportions, not learned attention or pixel-level correctness.
- The deployed classical branch is guided Retinex only; the full classical pipeline is not integrated.
- The UI's intermediate-result panels are placeholders.
- CPU inference latency is hardware-dependent; local timing does not establish hosted performance. `render.yaml` declares `plan: free` for both services, while `RENDER_DEPLOYMENT.md` says they request the `standard` plan and that a 512 MiB service is too small for the backend (522.9 MiB measured locally on a 256x256 image); this inconsistency is unresolved.
- The checkpoint and datasets are separately provisioned and are not included in Git.
- Dataset-specific results do not guarantee performance on all unseen scenes or sensors.

## 6. Deployment and Quickstart

Initialize the submodule:

```powershell
git submodule update --init --recursive
```

Provision and verify the separately distributed official checkpoint as described in `M3_SETUP_GUIDE.md` and `README.md` §6.

Start the API:

```powershell
python -m uvicorn deep_learning.api:app --host 127.0.0.1 --port 8000
```

Start the frontend in another terminal:

```powershell
python -m streamlit run app/main.py
```

Live Demo: https://lumina-streamlit.onrender.com/ — has been verified on 7 Nov 2026 from a laptop and a phone: the app loads and then enhances an uploaded image with the Official RetinexFormer mode. First load after inactivity may be slow due to free-tier cold start.

## 7. Validation and Reproducibility

The M3 final report records 129 passed tests, 13 warnings, and 7 subtests, along with AppTest, API, validation, and eval15 checks. These results must be rerun on the final release candidate. Use the fixed split manifest and commands in `evaluation/README.md` and `M3_EVALUATION_REPORT.md`; do not tune on eval15.

## 8. Team Contributions

| USN           | Member          | Implementation, testing, documentation, presentation |
| ------------- | --------------- | ---------------------------------------------------- |
| 1RUA24CSE0442 | Shreyank Balaji | Contributed to project                               |

                                                  implementation and
                                                  integration, testing
                                                  and validation,
                                                  preparation of project
                                                  documentation, and
                                                  presentation of the
                                                  solution.                                                 |

| 1RUA24CSE0472 | Sparsha G | Contributed to project
implementation, testing
and validation,
preparation of project
documentation, and
presentation of the
solution. |
| 1RUA24CSE0493 | Tanisha Shrikant Gavali | Contributed to project
implementation, testing
and validation,
preparation of project
documentation, and
presentation of the
solution. |
| 1RUA24CSE0512 | Vaishnavi Dhanuka | Led frontend
development and
integration of the
classical,
deep-learning, and
hybrid pipelines;
contributed to
deployment
coordination,
application testing,
evaluation, and
presentation
preparation. |
