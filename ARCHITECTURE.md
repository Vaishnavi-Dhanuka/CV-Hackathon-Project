# System Architecture

## 1. System overview

The system consists of a Streamlit frontend, a classical CV branch, a FastAPI-backed RetinexFormer branch, and an experimental hybrid fusion branch. The production recommendation from the M3 evaluation is standalone pretrained RetinexFormer.

## 2. Runtime data flow

```text
                       User uploads RGB image
                                |
                         Streamlit / LUMINA
                                |
              +-----------------+-----------------+
              |                 |                 |
          Original          Classical CV       DL / RetinexFormer
              |                 |                 |
          Preview         classical_client    dl_client
                                |                 |
                         M1 preprocessing     HTTP POST /enhance
                                |                 |
                         Guided Retinex       FastAPI service
                                |                 |
                                |          retinex_inference.py
                                |                 |
                                |          Official RetinexFormer
                                |                 |
                                +--------+--------+
                                         |
                               Experimental hybrid
                               spatial-adaptive fusion
                                         |
                              RGB output / explainability
                                         |
                                Preview and download
```

## 3. Frontend

`app/main.py` is the Streamlit entry point. It accepts an uploaded image, presents pipeline choices, stores results in session state, renders comparisons, and provides PNG downloads. The reviewed implementation exposes Original Image, Classical CV, Official RetinexFormer, Deep Learning, and Hybrid Intelligence modes. The app's Classical CV option runs guided Retinex only; it is not the separate full pipeline in `classical/`.

Image diagnostics such as brightness, contrast, sharpness, histograms, and noise proxies describe input characteristics. They are not substitutes for paired restoration metrics.

## 4. Classical CV branch

`hybrid/classical_client.py` adapts M1's `prepare_image()` and `retinex_enhance()` functions. It restores original dimensions when required and returns a contiguous RGB `uint8` array with a measured branch latency. The classical package and OpenCV contrib `ximgproc` module are required.

## 5. Deep-learning branch

`hybrid/dl_client.py` encodes an RGB image as PNG and sends it to the FastAPI `/enhance` endpoint. It validates the returned PNG and dimensions.

`deep_learning/api.py` accepts the multipart `file` upload, validates image size and decoded pixel limits, calls the inference function, and returns PNG output. The inference implementation in `deep_learning/retinex_inference.py` parses the official LOLv2-real configuration, strictly loads `checkpoint["params"]`, caches the model, pads to model-compatible dimensions, crops to the original dimensions, and returns RGB output.

The primary API checkpoint is:

`results/checkpoints/official_only/retinexformer_official_baseline.pth`

The documented checkpoint size is 6,478,393 bytes and SHA-256 is `539bd16c4da6179e45616329f249c4672951b1045193428e1d042c50d4b65a0b`.

## 6. Hybrid fusion

`hybrid/fusion.py` provides:

- `weighted_fusion()` â€” fixed scalar blend.
- `adaptive_fusion()` â€” global classical weight derived heuristically from input brightness, underexposure fraction, and luminance contrast.
- `spatial_adaptive_fusion()` â€” spatially varying, smoothed, bounded weights derived from low-light input luminance.

The same scalar weights are applied to RGB channels. The deep-learning map complements the classical map. These are heuristic blending proportions, not learned attention, SHAP explanations, or evidence that a branch is more correct. Fusion can dilute the stronger RetinexFormer result or alter colour and contrast.

The explainability component displays actual returned weight maps, image-wide mean contributions, and the global baseline. A contribution percentage describes the blend, not restoration quality.

## 7. Deployment topology

The Render Blueprint defines two web services:

- `lumina-retinex-api` â€” FastAPI backend.
- `lumina-streamlit` â€” Streamlit frontend.

The frontend receives the backend's external URL through `LUMINA_DL_API_URL`. The local client default is `http://127.0.0.1:8000`. Render configuration currently specifies the Free plan for both services; successful production performance on that plan has not been established by the local smoke tests.

## 8. Evaluation architecture

`evaluation/runner.py` evaluates paired low-light/reference images and reports PSNR, SSIM, LPIPS, NIQE, inference latency, and separate fusion latency. Validation experiments compare fusion candidates on a fixed split. The eval15 finalizer writes new outputs and avoids overwriting the pre-existing official eval15 CSV.

## 9. Limitations

- Classical CV depends on the M1 package and OpenCV contrib.
- Inference requires the separately provisioned checkpoint and initialized submodule.
- NIQE is a no-reference naturalness statistic, not reference fidelity.
- Fusion weights are heuristic.
- The M3 eval15 results show spatial fusion underperforming standalone RetinexFormer.
- Render cold-start, memory use, and public-service latency require deployment verification.

