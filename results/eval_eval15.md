# classical pipeline, eval15, 15 images

| config | PSNR | SSIM | latency ms | noise | edge corr | dPSNR vs FULL |
|---|---|---|---|---|---|---|
| input (no processing) | 7.77 | 0.191 | 0 | 1.22 | 0.833 | -10.20 |
| FULL pipeline | 17.97 | 0.765 | 199 | 2.38 | 0.888 | +0.00 |
| - adaptive Retinex | 19.03 | 0.756 | 200 | 2.84 | 0.869 | +1.05 |
| - color correction | 17.02 | 0.742 | 126 | 2.31 | 0.892 | -0.96 |
| - CLAHE | 17.97 | 0.766 | 190 | 2.32 | 0.889 | -0.01 |
| - denoise | 17.13 | 0.519 | 178 | 8.53 | 0.874 | -0.84 |
| - Retinex | 9.09 | 0.326 | 170 | 0.32 | 0.641 | -8.88 |
| Retinex only | 17.16 | 0.477 | 103 | 9.32 | 0.867 | -0.81 |
