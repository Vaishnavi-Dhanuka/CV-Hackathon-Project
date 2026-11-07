# M3 Evaluation Report

## Scope and protocol

This report records the seeded `our485` validation used to select a fusion
strategy and the subsequent one-shot `eval15` comparison. The `eval15` results
were not used to select or tune a strategy. All reference-based metrics compare
the enhanced result with its paired ground truth. If resizing is needed, the
runner preserves aspect ratio and uses Pillow Lanczos; mismatched aspect ratios
are left unscored and reported. NIQE is a no-reference naturalness statistic,
not a measure of ground-truth fidelity. Latency is not restoration quality.

The fixed split manifest is
[`evaluation/results/lolv2_our485_validation_seed2026.json`](./evaluation/results/lolv2_our485_validation_seed2026.json):
388 training names and 97 paired validation images from original LOL (LOL-v1) `our485`.
Validation compared Classical CV, the official RetinexFormer API, the existing
weighted/adaptive/spatial fusion implementations, and fixed RetinexFormer-
dominant blends. Each branch was inferred once per image and its output reused
for all fusion candidates.

## Validation results

Means over 97 paired validation images:

| Method | PSNR â†‘ | SSIM â†‘ | LPIPS â†“ | NIQE â†“ | CIEDE2000 â†“ | Fusion ms |
|---|---:|---:|---:|---:|---:|---:|
| Classical CV | 16.9612 | 0.4878 | 0.4394 | 8.2752 | 15.8826 | N/A |
| Pretrained Retinexformer | 22.1058 | 0.8451 | 0.1645 | 4.0858 | 8.4047 | N/A |
| Weighted 50:50 | 19.8884 | 0.6646 | 0.2922 | 6.2229 | 11.8449 | 11.73 |
| Adaptive fusion | 18.8833 | 0.5983 | 0.3457 | 7.0038 | 13.2636 | 18.15 |
| Spatial-adaptive fusion | 18.4391 | 0.5711 | 0.3678 | 7.3426 | 13.8660 | 49.93 |
| DL-dominant 90:10 | 21.8293 | 0.8286 | 0.1743 | 4.0949 | 8.7510 | 12.51 |
| DL-dominant 95:5 | 21.9894 | 0.8399 | 0.1671 | 4.0430 | 8.5234 | 12.57 |
| DL-dominant 80:20 | 21.4234 | 0.7932 | 0.1983 | 4.5403 | 9.3953 | 13.04 |

Fusion latency is separate from the sum of branch inference latencies. A
candidate was eligible only if the paired-bootstrap 95% lower bounds for PSNR,
SSIM, and `-LPIPS` were all positive, with no worse mean NIQE or mean CIEDE2000
than RetinexFormer. No candidate met that rule. The decision file therefore
selects **Pretrained Retinexformer** as the recommended production method;
fusion remains available as an experimental comparison.

The per-image records, summary, decision, color-fidelity data, and error log
are in `evaluation/results/` with the `lolv2_our485_fusion_validation_seed2026_verified`
prefix. All 97 images completed with all listed metrics valid and no recorded
validation failures.

## One-shot held-out eval15 results

After freezing the validation decision, the runner evaluated all 15 paired
eval15 images, producing 45 method rows. There were no inference or metric
failures.

| Method | PSNR â†‘ | SSIM â†‘ | LPIPS â†“ | NIQE â†“ | CIEDE2000 â†“ | Inference ms | Fusion ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| Classical CV | 16.3686 | 0.5048 | 0.3590 | 7.9804 | 16.2439 | 88.37 | N/A |
| Pretrained Retinexformer | 21.7720 | 0.8224 | 0.1524 | 3.8765 | 8.9765 | 1960.97 | N/A |
| Hybrid Intelligence (spatial-adaptive) | 18.1390 | 0.5885 | 0.2943 | 7.0345 | 13.7454 | 2395.17 | 47.51 |

Relative to RetinexFormer, the spatial Hybrid mean was lower by 3.6330 dB
PSNR and 0.2339 SSIM, and higher by 0.1419 LPIPS, 3.1580 NIQE, and 4.7690
CIEDE2000. It improved PSNR on 3 of 15 images, but did not improve mean
quality metrics. These measurements do not support claiming that Hybrid
outperforms RetinexFormer.

The generated files are:

- Per-image CSV: [`lolv2_eval15_m3_final_seed2026.csv`](./evaluation/results/lolv2_eval15_m3_final_seed2026.csv)
- Error log: [`lolv2_eval15_m3_final_seed2026.csv.errors.txt`](./evaluation/results/lolv2_eval15_m3_final_seed2026.csv.errors.txt)
- Aggregate charts and summary: `evaluation/results/lolv2_eval15_m3_final_seed2026_artifacts/charts/`
- Per-image deltas and color-fidelity CSVs: `evaluation/results/lolv2_eval15_m3_final_seed2026_artifacts/`
- Enhanced PNGs by method: `evaluation/results/lolv2_eval15_m3_final_seed2026_outputs/`
- Side-by-side least-/most-degraded examples (ranked only by Hybrid PSNR delta):
  `evaluation/results/lolv2_eval15_m3_final_seed2026_artifacts/`

The pre-existing `evaluation/results/lolv2_eval15_official_checkpoint.csv`
was not overwritten. Separately reported team test-set values are not copied
into these results.

## Reproduction

From the repository root, start the API and run the fixed validation and
one-shot finalizer using the commands in [the evaluation README](./evaluation/README.md).
The finalizer refuses existing output paths; choose unique paths for any new
run.

