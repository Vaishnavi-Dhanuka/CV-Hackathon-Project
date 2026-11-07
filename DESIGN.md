# Design Document

## 1. Problem

The project addresses low-light images with poor visibility, noise, colour distortion, and loss of detail. The target is to improve visibility while retaining image structure and avoiding misleading claims about restoration quality.

## 2. Design goals

1. Provide a usable image-upload and comparison interface.
2. Preserve a classical CV baseline.
3. Integrate an official pretrained RetinexFormer inference path.
4. Make experimental fusion available for comparison without presenting it as a proven improvement.
5. Support paired evaluation and reproducible reporting.
6. Handle invalid and oversized API inputs with explicit errors.

## 3. Main design choices

### 3.1 RetinexFormer as the production recommendation

The official pretrained RetinexFormer is used by the API. The model is loaded once, uses strict checkpoint state-dictionary loading, and selects CPU when CUDA is unavailable. The production checkpoint is stored outside Git and verified by size and SHA-256.

The M3 seeded validation did not identify a fusion candidate meeting the predeclared paired-bootstrap selection rule. The subsequent eval15 results also show lower mean quality metrics for spatial Hybrid than standalone RetinexFormer. Therefore the documented recommendation is standalone RetinexFormer.

### 3.2 Classical baseline retained

The classical branch adapts M1 preprocessing and guided Retinex enhancement. Keeping it provides a non-deep-learning comparison and allows the team to inspect differences between processing families.

### 3.3 Hybrid fusion retained as experimental

Fusion is deterministic and heuristic. The spatial map is derived from input luminance and bounded/smoothed. It is not trained as an attention mechanism and does not estimate per-pixel correctness. It remains useful as an experimental design comparison, not as the production quality winner.

### 3.4 Frontend/backend separation

The Streamlit UI communicates with a FastAPI endpoint through an HTTP client. This separates presentation from model inference and permits local or hosted backend URLs through `LUMINA_DL_API_URL`.

### 3.5 Input and output contracts

The reviewed adapters use RGB images and return RGB `uint8` arrays at original dimensions. The API accepts multipart image uploads and responds with PNG. The client validates response format and dimensions.

## 4. Error handling and limits

The API includes invalid-image handling, upload-size and decoded-pixel limits, missing-model handling, and a production-safe generic error response. The frontend client applies a request timeout and reports request/response errors to the application.

## 5. Evaluation design

The M3 validation used a fixed seeded 80/20 split of original LOL (LOL-v1) `our485`: 388 training names and 97 paired validation images. Candidate fusion strategies were compared against standalone methods. A paired-bootstrap eligibility rule required positive 95% lower bounds for PSNR, SSIM, and negative LPIPS, with no worse mean NIQE or CIEDE2000 than RetinexFormer.

After the validation decision was frozen, a one-shot eval15 comparison was run on 15 paired images, producing 45 method rows. The eval15 set was not used to select or tune a strategy.

## 6. Trade-offs

- RetinexFormer provides the strongest documented restoration metrics among the compared M3 methods, but CPU inference is substantially slower than the classical branch.
- Classical CV is fast in the reported eval15 run but has lower restoration metrics.
- Hybrid incurs both branch inference costs plus fusion and does not improve mean quality in the reported eval15 run.
- A hosted free-tier service may have cold-start and resource constraints; public deployment must be tested independently.

## 7. Known limitations

- Results are dataset- and run-specific and do not establish performance on every camera, scene, or unseen distribution.
- NIQE does not use paired ground truth and should not be interpreted as a direct fidelity score.
- Latency depends on hardware and service load.
- The local smoke tests do not establish Render deployment reliability.
