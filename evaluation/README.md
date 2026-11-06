# Paired-image evaluation runner

The evaluation runner applies the same set of low-light input images to three
M3 workflows:

- Classical CV (`hybrid.classical_client.enhance_with_classical`)
- Pretrained Retinexformer through the configured Deep Learning API
  (`hybrid.dl_client.enhance_with_api`)
- Hybrid Intelligence (both clients followed by
  `hybrid.fusion.spatial_adaptive_fusion`)

## Run

Provide the low-light input directory, its ground-truth directory, and an
output CSV path. The directories are supplied by the user; this project does
not assume or prescribe a dataset location.

```powershell
python -m evaluation.runner `
  --input-dir "path\to\low-light-images" `
  --reference-dir "path\to\ground-truth-images" `
  --output "path\to\results.csv"
```

The runner recursively scans supported image files (`.bmp`, `.jpeg`, `.jpg`,
`.png`, `.tif`, and `.tiff`). Inputs and references are paired by their
relative path and filename, so directory structures should match. The input
directory defines the test set; each input produces one CSV row per method.

The API method requires the M2 FastAPI server to be running at the URL
configured by `LUMINA_DL_API_URL` (default `http://127.0.0.1:8000`). The
`classical` package and the project dependencies used by the metric functions
must also be available in the Python environment.

## Metrics, timing, and resizing

PSNR, SSIM, and LPIPS use the existing paired-reference metric
implementations. NIQE uses the existing no-reference implementation.
Reference-based metrics run only when a matching, readable reference exists.
Latency is reported in milliseconds: inference latency for Classical CV and
Deep Learning is taken from each client; Hybrid `latency_ms` is the sum of its
two sequential enhancement client latencies. Hybrid `fusion_latency_ms` is
measured separately around the fusion call and is not included in that sum.

Enhanced results are compared at their output dimensions. When the ground
truth has the same aspect ratio, it is resized to those dimensions with Pillow
Lanczos resampling. Images with different aspect ratios are not stretched or
cropped; their paired metrics are left blank and the mismatch is reported.
This keeps spatial proportions intact and avoids scoring misaligned geometry.
Any resizing affects the reference used for metrics, not the input sent to an
enhancement method.

## Failures

Inference and metric failures do not produce guessed values. The CSV keeps
empty cells for unavailable metrics, but still records one row per input and
method. A sibling `<output>.errors.txt` lists missing or unreadable references,
failed inference, aspect-ratio mismatches, and metric errors. The CLI prints
the issues and exits with status 1 when evaluation produced per-image
failures; setup errors exit with status 2. Check both the CSV and the error
report before comparing methods.

The CSV columns follow `results_template.csv`:
`image_name, method, psnr, ssim, lpips, niqe, latency_ms, fusion_latency_ms,
output_height, output_width`.

## Visualize results

After producing an evaluation CSV, aggregate its metrics and create available
comparison charts:

```powershell
python -m evaluation.visualize_results `
  --input "path\to\results.csv" `
  --output-dir "path\to\visualizations"
```

The output directory contains individual PNG charts for PSNR, SSIM, LPIPS,
NIQE, inference latency, and Hybrid fusion latency when that metric has at
least one valid observation. Bars show method means, error bars show
population standard deviation, and labels include the number of valid
samples. Missing, blank, NaN, and non-finite values (except positive-infinite
PSNR, which can represent an exact match) are excluded rather than filled with
zero. Metrics with no valid observations have summary rows with `n=0` and no
chart.

The module also writes `summary.csv` with mean, median, standard deviation,
and valid sample count for each metric/method pair, plus a concise
`summary.txt`. Aggregates describe the supplied results only; the summary
does not infer benchmark quality or treat latency as restoration accuracy.

## Hybrid fusion contribution maps

When Hybrid Intelligence completes in the Streamlit app, the comparison view
also displays the spatial Classical CV and Deep Learning fusion weight maps,
a signed Classical-minus-DL contribution heatmap, and the global baseline
weights. The maps show the fusion algorithm's normalized heuristic
contribution weights: higher Classical CV weight means greater Classical CV
contribution, while higher DL weight means greater Deep Learning contribution.
An optional checkbox overlays the relative contributions on the original
image.

These are deterministic heuristic fusion contribution maps, not model
attention maps, learned explanations, or evidence that either restoration is
better. The visualizer validates map ranges and complementary weights and
displays the returned values on a fixed [0, 1] scale without changing fusion
calculations.

## Validation-only fusion experiments

Fusion selection uses paired images from `data/LOLdataset/our485/low` and
`data/LOLdataset/our485/high`. The runner creates a deterministic 80/20 split
with seed 2026, keeps input/reference pairs together, and writes the relative
filenames in a JSON manifest. It refuses to generate a validation split from
`eval15`. Do not change the manifest after seeing eval15 results.

Create the manifest once, then run all candidate strategies on the 97-image
validation set. Start the FastAPI server before the full run:

```powershell
python -m evaluation.fusion_experiments `
  --input-dir data/LOLdataset/our485/low `
  --reference-dir data/LOLdataset/our485/high `
  --manifest evaluation/results/lolv2_our485_validation_seed2026.json `
  --output evaluation/results/lolv2_our485_fusion_validation_seed2026_verified.csv `
  --create-split-only

python -m evaluation.fusion_experiments `
  --input-dir data/LOLdataset/our485/low `
  --reference-dir data/LOLdataset/our485/high `
  --manifest evaluation/results/lolv2_our485_validation_seed2026.json `
  --output evaluation/results/lolv2_our485_fusion_validation_seed2026_verified.csv
```

The experiment compares Classical CV, RetinexFormer, 50:50 weighted fusion,
adaptive fusion, spatial-adaptive fusion, and fixed 90:10, 95:5, and 80:20
RetinexFormer-dominant blends. It runs each enhancement branch once per image
and reuses those branch outputs for all fusion candidates. Outputs include the
per-image method CSV, candidate summary CSV, paired-bootstrap selection JSON,
CIEDE2000 color-fidelity CSV, and an error log. The selection rule is recorded
in the JSON and requires consistent paired PSNR, SSIM, and LPIPS improvements,
no worse mean NIQE, and no worse mean CIEDE2000 color difference. NIQE is
descriptive and is not treated as reference fidelity.

## One-shot held-out eval15 run

Only after the validation selection is fixed, run the held-out split once.
The finalizer saves the standard per-image results CSV, the sibling error log,
charts/aggregate CSV, per-image deltas against RetinexFormer, CIEDE2000
color-fidelity results, all enhanced PNGs, and best/least-degraded and
worst/most-degraded side-by-side examples. Provide new output paths; existing
benchmark files are never replaced.

```powershell
python -m evaluation.finalize_benchmark `
  --input-dir data/LOLdataset/eval15/low `
  --reference-dir data/LOLdataset/eval15/high `
  --output evaluation/results/lolv2_eval15_m3_final_seed2026.csv `
  --artifacts-dir evaluation/results/lolv2_eval15_m3_final_seed2026_artifacts `
  --enhanced-images-dir evaluation/results/lolv2_eval15_m3_final_seed2026_outputs
```

The finalizer evaluates the three deployed workflows. Its `Hybrid Intelligence`
row represents the current spatial-adaptive fusion pipeline; if validation
selects RetinexFormer alone, that is the recommended production output and the
spatial fusion result remains an experimental comparison, not a claimed win.
These paths are the completed one-shot run; select new output paths before
starting another run. The measured validation decision and eval15 results are
documented in [M3_EVALUATION_REPORT.md](../M3_EVALUATION_REPORT.md).

## Recorded M3 results

The fixed seed-2026 validation split contained 97 paired images. No tested
fusion strategy passed the paired-bootstrap improvement requirements, so
RetinexFormer alone was selected. On the separate 15-image eval15 run,
RetinexFormer averaged 21.7720 PSNR, 0.8224 SSIM, 0.1524 LPIPS, and 3.8765
NIQE; the spatial Hybrid baseline averaged 18.1390 PSNR, 0.5885 SSIM, 0.2943
LPIPS, and 7.0345 NIQE. The eval15 findings do not support claiming that
Hybrid outperforms RetinexFormer. See the report for the full metrics,
latencies, artifact paths, and limitations.
