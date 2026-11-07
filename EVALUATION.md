# Evaluation and Reproducibility

## 1. Evaluation scope

The M3 report records a seeded `our485` validation used to select a fusion strategy and a subsequent one-shot `eval15` comparison. The eval15 results were not used to select or tune a strategy.

Reference-based metrics compare each enhanced output with its paired ground truth. If resizing is needed, the runner preserves aspect ratio and uses Pillow Lanczos; mismatched aspect ratios are left unscored and reported.

## 2. Metrics

- **PSNR (dB):** reference-based pixel error metric; higher is better.
- **SSIM:** reference-based structural similarity; higher is better.
- **LPIPS:** learned perceptual distance; lower is better.
- **NIQE:** no-reference naturalness statistic; lower is conventionally preferred, but it is not a ground-truth fidelity measure.
- **CIEDE2000:** colour-difference diagnostic; lower indicates less colour difference.
- **Inference latency:** measured branch inference time; not an image-quality metric.
- **Fusion latency:** measured separately from branch inference.

## 3. Seeded validation protocol

The fixed split manifest is:

`evaluation/results/lolv2_our485_validation_seed2026.json`

It contains 388 training names and 97 paired validation images from original LOL (LOL-v1) `our485`. The compared methods were Classical CV, official RetinexFormer, weighted/adaptive/spatial fusion, and fixed RetinexFormer-dominant blends. Each branch was inferred once per image and reused for fusion candidates.

### Validation means over 97 paired images

| Method | PSNR | SSIM | LPIPS | NIQE | CIEDE2000 | Fusion ms |
|---|---:|---:|---:|---:|---:|---:|
| Classical CV | 16.9612 | 0.4878 | 0.4394 | 8.2752 | 15.8826 | N/A |
| Pretrained RetinexFormer | 22.1058 | 0.8451 | 0.1645 | 4.0858 | 8.4047 | N/A |
| Weighted 50:50 | 19.8884 | 0.6646 | 0.2922 | 6.2229 | 11.8449 | 11.73 |
| Adaptive fusion | 18.8833 | 0.5983 | 0.3457 | 7.0038 | 13.2636 | 18.15 |
| Spatial-adaptive fusion | 18.4391 | 0.5711 | 0.3678 | 7.3426 | 13.8660 | 49.93 |
| DL-dominant 90:10 | 21.8293 | 0.8286 | 0.1743 | 4.0949 | 8.7510 | 12.51 |
| DL-dominant 95:5 | 21.9894 | 0.8399 | 0.1671 | 4.0430 | 8.5234 | 12.57 |
| DL-dominant 80:20 | 21.4234 | 0.7932 | 0.1983 | 4.5403 | 9.3953 | 13.04 |

A candidate was eligible only if paired-bootstrap 95% lower bounds for PSNR, SSIM, and `-LPIPS` were all positive, with no worse mean NIQE or CIEDE2000 than RetinexFormer. No fusion candidate met the rule. The validation decision selected pretrained RetinexFormer.

## 4. One-shot held-out eval15

After freezing the validation decision, the runner evaluated 15 paired images, producing 45 method rows. The report records no inference or metric failures.

| Method | PSNR | SSIM | LPIPS | NIQE | CIEDE2000 | Inference ms | Fusion ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| Classical CV | 16.3686 | 0.5048 | 0.3590 | 7.9804 | 16.2439 | 88.37 | N/A |
| Pretrained RetinexFormer | 21.7720 | 0.8224 | 0.1524 | 3.8765 | 8.9765 | 1960.97 | N/A |
| Hybrid Intelligence (spatial-adaptive) | 18.1390 | 0.5885 | 0.2943 | 7.0345 | 13.7454 | 2395.17 | 47.51 |

Relative to RetinexFormer, spatial Hybrid was lower by 3.6330 dB PSNR and 0.2339 SSIM, and higher by 0.1419 LPIPS, 3.1580 NIQE, and 4.7690 CIEDE2000. It improved PSNR on 3 of 15 images but did not improve mean quality metrics.

## 5. Interpretation

The reported M3 results support standalone pretrained RetinexFormer as the production recommendation. They do not support claiming that the current spatial Hybrid improves restoration quality.

Do not combine these M3 results with other separately reported experiments:
- 69-image architecture comparison.
- Model A/B validation.
- 100-image official LOLv2 test.
- 97-image M3 validation.
- 15-image M3 eval15.

Keep dataset, split, checkpoint, and protocol labels attached to every table.

## 6. Reproduction checklist

- [ ] Initialize the pinned RetinexFormer submodule.
- [ ] Provision and hash-verify the exact official checkpoint.
- [ ] Verify dataset files and paired low-light/reference correspondence.
- [ ] Use the fixed validation manifest and recorded seed.
- [ ] Run validation candidate evaluation without using eval15.
- [ ] Freeze the validation selection decision.
- [ ] Run eval15 once with unique output paths.
- [ ] Preserve raw per-image CSVs, aggregate summaries, error logs, charts, and generated examples.
- [ ] Record Python/package versions, hardware, device, checkpoint hash, commands, and timestamps.
- [ ] Re-run the final evaluation on the release candidate and compare outputs.

## 7. Artifact locations

The reports identify artifacts under `evaluation/results/`, including the fixed split manifest, validation outputs with the `lolv2_our485_fusion_validation_seed2026_verified` prefix, and eval15 outputs with the `lolv2_eval15_m3_final_seed2026` prefix. Confirm these files exist in the final repository/worktree before claiming that all artifacts are included.

