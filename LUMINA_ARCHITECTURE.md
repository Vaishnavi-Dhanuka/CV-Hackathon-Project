# M3 Architecture

## Runtime data flow

```text
Uploaded RGB image
       |
       +--> Original Image -------------------------------> Download/display
       |
       +--> Classical CV adapter --> M1 preprocessing --> M1 guided Retinex
       |                                                   |
       +--> Official RetinexFormer adapter --> HTTP API --+--> display
       |                                      |            |
       +--> Hybrid Intelligence: both branches ----------> spatial fusion
                                                              |
                                                       fused RGB uint8
                                                              |
                                                  LUMINA display/download
```

### Input and frontend

`app/main.py` validates and analyzes the upload, stores result images in
Streamlit session state, and renders the comparison cards. The sidebar selects
Original Image, Classical CV, Official RetinexFormer, or Hybrid Intelligence.
Results are cached by the uploaded content hash and selected mode; standalone
branch outputs are reused when switching into Hybrid mode. PNG output bytes
drive both preview and download.

Input brightness, contrast, sharpness, histograms, and noise proxy are image
diagnostics. They do not measure restoration accuracy.

### Classical CV branch

`hybrid/classical_client.py` adapts M1's `prepare_image()` and default guided
`retinex_enhance()` functions. It restores the original dimensions after M1
preprocessing and returns a contiguous RGB `uint8` array with latency.
The `classical` package and OpenCV contrib's `ximgproc` module must be
available.

### Official deep-learning branch

`hybrid/dl_client.py` sends an in-memory RGB PNG to the FastAPI `/enhance`
endpoint and validates the returned PNG. The API delegates to the official
`deep_learning/retinex_inference.py`; it does not construct an alternate model.
That implementation parses the official LOLv2-real YAML, strictly loads
`checkpoint["params"]`, caches the model once, pads to model-compatible
dimensions, crops back to the original dimensions, and returns RGB PIL output.
The architecture is RetinexFormer and the only production checkpoint is
`results/checkpoints/official_only/retinexformer_official_baseline.pth`.

### Fusion and explainability

`hybrid/fusion.py` provides:

- `weighted_fusion()`: fixed scalar pixel blend.
- `adaptive_fusion()`: global Classical weight derived from input brightness,
  underexposure fraction, and luminance contrast.
- `spatial_adaptive_fusion()`: smooth, bounded spatial weight maps based only
  on low-light input luminance; it adjusts the global Classical weight by a
  small illumination prior and sets the DL map to its complement.

The same scalar weights apply to all RGB channels. The blend rounds and clips
to the uint8 range. These heuristics do not measure which branch is correct.
They can dilute the better RetinexFormer result, and blending distinct branch
outputs may alter color or contrast.

`app/components/fusion_explainability.py` validates and displays the actual
per-pixel maps, their image-wide mean contribution percentages, and the global
baseline. Each Classical/DL pair sums to 100%. The visualizations describe
heuristic blend proportions, not learned attention, SHAP, or quality gains.

### Evaluation

`evaluation/runner.py` evaluates Classical CV, RetinexFormer, and spatial
Hybrid on paired low-light/reference images. It reports PSNR, SSIM, LPIPS,
NIQE, inference latency, and separate fusion latency. The validation
experiment uses a fixed paired our485 split to compare multiple fusion
strategies. The eval15 finalizer writes a new benchmark CSV, aggregate charts,
per-image deltas, color-fidelity diagnostics, errors, and visual examples
without overwriting previous results.

## Limitations

- Classical CV requires the team's M1 `classical` package.
- The API and Streamlit frontend require the separately provisioned checkpoint.
- NIQE is a no-reference naturalness statistic, not reference fidelity.
- Fusion weights are heuristics; the existing eval15 benchmark shows that
  spatial fusion can underperform RetinexFormer.
- CIEDE2000 is reported as a color-difference diagnostic; it does not replace
  PSNR, SSIM, or LPIPS.
