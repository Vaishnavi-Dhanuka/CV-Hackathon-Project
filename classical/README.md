# Classical CV pipeline: low-light restoration (P1)

Member 1 module of LUMINA. `enhance_classical(image, config=None) -> (enhanced, intermediate)`.
Input and output are uint8 RGB. `intermediate` holds illumination, reflectance, retinex, clahe, denoised, final, plus a `meta` dict (per-stage latency, config, measured image statistics).

## Pipeline

| Stage                       | What it does                                                                                                                            | Why                                                                       |
| --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| Preprocessing               | Downscale to max side 1024, BGR to RGB once at load                                                                                     | Bounded latency, consistent colour handling                               |
| Illumination                | Guided-filter estimate of L from max(R,G,B), decomposition into illumination and reflectance                                            | Retinex theory: image = reflectance x illumination                        |
| Adaptive Retinex parameters | Mean brightness picks one of 3 bins, each with its own target mean and gamma                                                            | Dark and moderately dark images need different gain                       |
| Guided Retinex              | I x L^(gamma-1), brightness restore, soft clip, blend                                                                                   | Lifts shadows while keeping edges (guided filter, no halos)               |
| Guided denoise              | sigma = 6 x measured noise, clipped to 25-70                                                                                            | Retinex amplifies noise, so smoothing strength follows the measured noise |
| CLAHE                       | clip 1.0, 8x8 tiles, strength 0.5, only if contrast after denoise < 100                                                                 | Local contrast, applied only when the image is flat                       |
| Colour correction           | Gray-world white balance (strength ramps 0.05-0.20 with measured cast), adaptive gamma (0.8-1.1), highlight roll-off on the max channel | Remove casts and avoid blown highlights while preserving hue              |

Contrast stretch is implemented but disabled (`CONTRAST_STRENGTH = 0.0`): on our485 it cost -1.37 dB on average and -14 dB in the worst image.

## Data and tuning rule

- Parameters were tuned only on LOL **our485**. The adaptive rule was fitted on 97 our485 images and verified on 97 different ones.
- LOL **eval15** and **LOL-v2 Real test** (100 pairs) were never used for tuning. They are reported as verification only.
- Metrics: uint8 RGB, `data_range=255`, no brightness matching, all pairs, mean over images.

## Results

| Set                          | Input PSNR / SSIM | Classical PSNR / SSIM | Latency (CPU, mean) |
| ---------------------------- | ----------------- | --------------------- | ------------------- |
| LOL-v2 Real test (100 pairs) | 9.72 dB / 0.196   | **20.47 dB / 0.784**  | 169 ms (p95 198 ms) |
| LOL eval15 (15 pairs)        | not recorded here | **17.97 dB / 0.765**  | 199 ms              |

The classical output beats the raw input on 100 of 100 LOL-v2 images. eval15 has only 15 images, so differences of about 1 dB are within noise.

### Latency per stage

Mean per stage over 20 LOL-v2 images (400x600, CPU, `tests/stage_latency.py`):

| Stage                     | Mean (ms) | p95 (ms)  |
| ------------------------- | --------- | --------- |
| Preprocess                | 17.7      | 21.6      |
| Illumination              | 40.6      | 45.5      |
| Adaptive parameter choice | 0.0       | 0.0       |
| Retinex                   | 26.9      | 30.4      |
| Denoise                   | 23.7      | 30.2      |
| CLAHE                     | 0.0       | 0.0       |
| Colour correction         | 56.2      | 63.5      |
| **Total**                 | **167.6** | **187.5** |

Over all 100 LOL-v2 images (`_run_log.csv`) the mean total is 165.5 ms (min 125.3, max 343.6). The adaptive choice is a lookup and costs nothing measurable. CLAHE shows 0.0 ms because it did not run on these images (it only runs when contrast after denoising is below 100). Colour correction is the slowest stage, at about a third of the total, yet it changes PSNR the least in the ablation. Timings vary by a few ms between runs, so the totals in this README (165-170 ms) differ slightly.

## Ablation

Change in PSNR when the full pipeline is compared with the variant without the stage (FULL minus variant; positive means the stage helps).

| Stage removed               | LOL-v2 (dB) | eval15 (dB) | SSIM change (LOL-v2 / eval15) |
| --------------------------- | ----------- | ----------- | ----------------------------- |
| Retinex                     | +9.14       | +8.88       | not recorded                  |
| Denoise                     | +2.03       | +0.84       | +0.27 / +0.25                 |
| Adaptive Retinex parameters | +2.95       | -1.05       | not recorded                  |
| Colour correction           | -0.24       | +0.96       | +0.013 / +0.023               |
| CLAHE                       | about 0     | about 0     | not recorded                  |

Reading the table honestly:

- Retinex and denoising carry the result.
- CLAHE contributes about nothing on both sets. It is gated to flat images and rarely fires.
- Colour correction slightly hurts PSNR on LOL-v2 but improves SSIM on both sets.
- The "- adaptive" variant falls back to the old fixed default (target mean 0.5). So part of the +2.95 dB on LOL-v2 may come from the lower target mean 0.4 and not from per-image adaptation. We have not separated the two. The adaptive rule also hurts on eval15 (-1.05 dB, 15 images).

## Adaptive rule

Feature: mean brightness, bins at 11.21 and 16.07. Bin 0: target mean 0.4, gamma 0.5. Bins 1-2: target mean 0.4, gamma 0.7. On the 97-image check set from our485: 19.56 dB / 0.768 vs 19.25 dB for the fixed default (+0.31 dB). Illumination CV and dark fraction were tested and were not better. The best per-image choice reached about 21.7 dB, so the rule captures only part of the available gain.

## Failure analysis

**1. Exposure target (LOL-v2 low00779, 00780, 00770: 13.6-13.8 dB).** Input mean about 10.5, output mean about 104-108, ground truth mean about 58-63, so these are over-exposed by about 45 gray levels. Reference brightness differs by dataset: our485 references average 116.1, eval15 114.9 and LOL-v2 94.0. Our outputs on LOL-v2 average 105.0 with a standard deviation of 8.2 across images, against 19.0 for the references. The pipeline pulls every image toward one brightness, so scenes that are meant to stay dark get over-brightened. A fixed target cannot know the intended exposure. We did not tune the target on test data. Known inconsistency: the colour-correction gamma targets mean 0.5 while Retinex targets 0.4, so the two stages partly fight.

**2. Unseen phone photos.** Visual inspection of 16 own photos (no ground truth):

- Extremely dark inputs (img1, img4) become flat gray with blotchy patches, because noise is amplified and then over-smoothed.
- Warm-lit scenes (img3, img9, img10, img11, img12, img16) get a blue or purple cast. Gray-world white balance assumes the scene averages to gray, which is false in a lamp-lit room, so the lamp in img11 shifts from warm to cyan.
- Bright sources cause halos and blown highlights (img16).
- Causes in this list are hypotheses and are not yet isolated by ablation.
- img6, img13, img14 and img15 were already well exposed and are controls. The pipeline left them nearly unchanged.

## Unseen set

16 phone photos taken by the team, downscaled once to max side 1024 (INTER_AREA) and saved as PNG, so all approaches see identical inputs. No ground truth exists, so only latency, NIQE and visual inspection apply.

## Reproduce

```
pip install -r requirements.txt        # opencv-contrib-python (needs cv2.ximgproc), numpy, scikit-image, matplotlib
python -m tests.dump_outputs <input_folder> <output_folder>
python -m tests.prepare_wild data\wild_original data\wild
python -m tests.gt_brightness <label> <folder> [<label> <folder> ...]
```

python -m tests.evaluate data\LOLv2\Test\Low data\LOLv2\Test\Normal lolv2
python -m tests.evaluate <eval15\low> <eval15\high> eval15

Python 3.13, CPU only. Latency about 170-200 ms per 400x600 image.

### Component demos and smoke checks

The image demos and input smoke check are in `scripts/classical_demos/` so
pytest does not mistake their command-line arguments and import-time behavior
for tests. Run them with the repository on `PYTHONPATH`. The demos write
relative `results/*.png` outputs under the current working directory; use a
temporary working directory to keep generated files away from tracked results.

From the repository root in PowerShell:

```powershell
$repo = (Get-Location).Path
$work = Join-Path $env:TEMP ("cv-classical-demos-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $work | Out-Null
$env:PYTHONPATH = $repo
$env:MPLBACKEND = "Agg"
Push-Location $work

python "$repo\scripts\classical_demos\clahe_demo.py" "$repo\data\LOLdataset\our485\low\10.png"
python "$repo\scripts\classical_demos\color_correction_demo.py" "$repo\data\LOLdataset\our485" 485
python "$repo\scripts\classical_demos\denoising_demo.py" "$repo\data\LOLdataset\our485\high\10.png"
python "$repo\scripts\classical_demos\illumination_demo.py" "$repo\data\LOLdataset\our485\low\10.png"
python "$repo\scripts\classical_demos\preprocessing_demo.py" "$repo\data\LOLdataset\our485\low\10.png"
python "$repo\scripts\classical_demos\retinex_demo.py" "$repo\data\LOLdataset\our485\low\10.png"
python "$repo\scripts\classical_demos\bad_inputs_smoke.py"

Pop-Location
```

For pytest, run the actual test suites, for example:

```powershell
python -m pytest -p no:cacheprovider evaluation/tests -q
```

## Limitations

- Fixed exposure target (see failure 1); dataset-specific bias towards our485 brightness.
- Gray-world white balance fails under strongly coloured lighting.
- Extremely dark inputs lose detail after denoising.
- CLAHE has no measurable effect; the adaptive gain is partly confounded with the lower target mean.
- eval15 has 15 images, so conclusions from it are weak.
- No ground truth for the unseen photos, so no PSNR or SSIM there.
- Classical methods cannot recover information that sensor noise has destroyed. Deep models do better on LOL-v2 (see the team comparison).
- Bad inputs: images smaller than 16 px on a side and missing or unreadable files raise a clear `ValueError`. Grayscale and RGBA images are converted to RGB. Images larger than 1024 px on the long side are downscaled, so the output size can differ from the input size. The app should catch `ValueError` and show the message.

## Ethics

- The unseen photos are our own, contain no faces, and had EXIF/GPS data removed by conversion to PNG. Original files are not committed.
- LOL, LOL-v2 and eval15 data are not committed and are used under their research terms.
- Enhanced images are not evidence. The pipeline changes colours and amplifies noise, so its output must not be used for forensic or identification purposes.
- Early commits on this branch contain LOL dataset images. They were removed from the tree and are used under the dataset's research terms.
