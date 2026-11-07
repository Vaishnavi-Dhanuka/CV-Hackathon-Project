# classical pipeline, lolv2_real, 100 images

| config | PSNR | SSIM | latency ms | noise | edge corr | dPSNR vs FULL |
|---|---|---|---|---|---|---|
| input (no processing) | 9.72 | 0.196 | 0 | 1.52 | 0.818 | -10.75 |
| FULL pipeline | 20.47 | 0.784 | 169 | 3.06 | 0.875 | -0.00 |
| - adaptive Retinex | 17.52 | 0.744 | 167 | 3.70 | 0.841 | -2.95 |
| - color correction | 20.71 | 0.771 | 105 | 2.98 | 0.886 | +0.24 |
| - CLAHE | 20.62 | 0.783 | 165 | 2.96 | 0.875 | +0.14 |
| - denoise | 18.44 | 0.514 | 156 | 11.09 | 0.839 | -2.03 |
| - Retinex | 11.34 | 0.338 | 131 | 0.44 | 0.628 | -9.14 |
| Retinex only | 16.72 | 0.459 | 91 | 11.90 | 0.837 | -3.75 |
