1RUA24CSE0442, 1RUA24CSE0472, 1RUA24CSE0493, 1RUA24CSE0512

# Comparative Evaluation Results — P1, Team T21

## Evaluation setup

The comparison below reports results on the 15-image held-out LOL (v1) `eval15` split. The evaluation was performed after the fusion-selection decision was frozen. The three methods are the deployed classical adapter (guided Retinex only), pretrained RetinexFormer, and the spatial-adaptive hybrid.

## Comparative metrics

| Approach / Model Variant | PSNR (dB) ↑ | SSIM ↑ | Perceptual Metric (LPIPS / NIQE) ↓ | Latency (ms / frame) |
| :--- | ---: | ---: | :---: | ---: |
| Classical CV (Retinex only) | 16.3686 | 0.5048 | 0.3590 / 7.9804 | 88.37 |
| Deep Learning Only (Pretrained RetinexFormer) | 21.7720 | 0.8224 | 0.1524 / 3.8765 | 1960.97 |
| Proposed Hybrid System (Spatial-adaptive) | 18.1390 | 0.5885 | 0.2943 / 7.0345 | 2395.17 |

*Latency is the recorded inference latency. The hybrid additionally recorded 47.51 ms/frame of fusion overhead. Classical and standalone RetinexFormer have no fusion stage.*

## Comparative findings

- Pretrained RetinexFormer achieved the strongest average results across the reported quality metrics: the highest PSNR and SSIM, and the lowest LPIPS and NIQE.
- The spatial-adaptive hybrid did not outperform standalone RetinexFormer. Relative to RetinexFormer, its mean PSNR was lower by 3.6330 dB and SSIM was lower by 0.2339; LPIPS was higher by 0.1419 and NIQE was higher by 3.1580.
- The classical Retinex-only branch had the lowest recorded inference latency, but its average image-quality metrics were also the weakest of the three approaches.
- Based on this comparison, standalone pretrained RetinexFormer is the preferred quality-focused method. The hybrid remains experimental.

## Metric interpretation and scope

PSNR and SSIM are higher-is-better metrics. LPIPS and NIQE are lower-is-better metrics; NIQE is a no-reference statistic and should not be interpreted as a ground-truth fidelity measure. Latency is reported separately from image quality.

These figures describe the 15-image LOL (v1) `eval15` comparison only. They should not be mixed with the separate 97-image seeded validation results or the separate 100-image LOL-v2 Real test results.

## Reproducibility source

Per-image results: `evaluation/results/lolv2_eval15_m3_final_seed2026.csv`  
Summary: `evaluation/results/.../charts/summary.txt`  
Evaluation protocol and commands: `evaluation/README.md` and `M3_EVALUATION_REPORT.md`

The repository's per-image CSV is the source for the reported aggregate comparison.
