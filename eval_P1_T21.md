1RUA24CSE0442, 1RUA24CSE0472, 1RUA24CSE0493, 1RUA24CSE0512

# P1 - Robust Low-Light Image Restoration and Enhancement

## Team Information

| Field | Details |
|---|---|
| Team ID | T21 |
| Problem ID | P1 |
| Problem Title | Robust Low-Light Image Restoration and Enhancement |
| Team Name | VisionForge |
| Team Members | Shreyank Balaji; Sparsha G; Tanisha Shrikant Gavali; Vaishnavi Dhanuka |
| GitHub Repository | https://github.com/Vaishnavi-Dhanuka/CV-Hackathon-Project |
| Live Demo | [ADD PUBLIC DEPLOYMENT URL] |

### Team Members

| USN | Full Name |
|---|---|
| 1RUA24CSE0442 | Shreyank Balaji |
| 1RUA24CSE0472 | Sparsha G |
| 1RUA24CSE0493 | Tanisha Shrikant Gavali |
| 1RUA24CSE0512 | Vaishnavi Dhanuka |

---

# 1. Problem Understanding and Solution Strategy

## 1.1 Problem Understanding

The problem requires an intelligent illumination-adaptive system capable of enhancing severely underexposed, noisy and colour-distorted images while preserving fine textures, edges and structural information. The solution must estimate and improve illumination, suppress noise, preserve colour fidelity and recover structural details without excessive overexposure, halos or unnatural colours.

The project addresses the three required approaches: classical computer vision, deep-learning-based restoration, and hybrid intelligence combining classical and deep-learning outputs.

## 1.2 Proposed Solution

The project implements **LUMINA**, a low-light image restoration application with a Streamlit interface and a FastAPI-based deep-learning inference service.

The application provides Original Image, Classical CV, Official RetinexFormer, Deep Learning and Hybrid Intelligence options. The Deep Learning interface option is routed through the RetinexFormer API. The classical branch uses M1 preprocessing and guided Retinex enhancement. The deep-learning branch uses the official pretrained RetinexFormer model. The hybrid branch runs both enhancement paths and combines their outputs using spatially adaptive fusion.

Fusion was evaluated against standalone RetinexFormer. Since experiments did not establish reliable quality improvements from fusion, standalone RetinexFormer is the recommended production method.

---

# 2. Technical Architecture and Implementation

## 2.1 System Architecture

The system consists of:
- **Frontend:** Streamlit application for image upload, enhancement selection, visual comparison, metrics and PNG download.
- **Classical enhancement:** M1 preprocessing and guided Retinex enhancement.
- **Deep-learning inference:** FastAPI service serving the pretrained RetinexFormer model.
- **Hybrid fusion:** Spatially adaptive fusion of classical and deep-learning outputs.
- **Evaluation:** Paired-reference metrics, no-reference quality metrics, latency measurements and fusion ablation experiments.

### Processing Flow

```text
                User uploads image
                       |
                       v
              Streamlit Application
                       |
          +------------+-------------+
          |            |             |
          v            v             v
      Classical     RetinexFormer   Hybrid Mode
       Branch        API Branch        |
          |            |          +----+----+
          |            |          |         |
          |            |          v         v
          |            |      Classical  RetinexFormer
          |            |      Output      Output
          |            |          |         |
          |            |          +----+----+
          |            |               |
          |            |               v
          |            |       Spatial Adaptive Fusion
          |            |               |
          +------------+---------------+
                       |
                       v
             Enhanced Image and Evaluation
                       |
                       v
             Comparison and PNG Download
```

## 2.2 Classical Computer Vision

The classical branch uses M1 preprocessing followed by guided Retinex enhancement. It aims to improve illumination and local contrast while retaining image structure. The implementation includes image preprocessing, Retinex-based enhancement, guided filtering-based processing, and output validation/resizing where necessary.

## 2.3 Deep Learning

The deep-learning branch uses the official pretrained **RetinexFormer** architecture. The inference service loads the official configuration and selected checkpoint, processes uploaded RGB images, handles image dimensions through padding and cropping, and returns enhanced RGB images in PNG format.

The API is implemented using FastAPI.

| Endpoint | Purpose |
|---|---|
| GET `/` | API running information |
| GET `/health` | Checks model configuration and checkpoint availability |
| POST `/enhance` | Receives an image and returns the enhanced PNG |

Primary inference checkpoint:

`results/checkpoints/official_only/retinexformer_official_baseline.pth`

## 2.4 Hybrid Intelligence and Adaptive Fusion

The hybrid pipeline executes the classical and deep-learning branches and combines their outputs. Spatial-adaptive fusion uses image luminance information to construct a spatially varying contribution map. It computes a global baseline contribution, estimates local illumination, adjusts the classical contribution according to local illumination, smooths the contribution map, constrains fusion weights, and uses complementary classical and deep-learning weight maps.

The fusion method is heuristic and deterministic. It is not a learned attention mechanism and does not directly measure which output has better restoration quality at each pixel. The experiments showed that the tested fusion strategies did not reliably improve restoration quality over standalone RetinexFormer.

## 2.5 Evaluation Metrics

- **PSNR:** Pixel-level reconstruction quality.
- **SSIM:** Structural similarity.
- **LPIPS:** Perceptual similarity.
- **NIQE:** No-reference image-quality indicator.
- **CIEDE2000:** Colour difference.
- **Latency:** Processing time in milliseconds.

PSNR, SSIM, LPIPS and CIEDE2000 use paired reference images. NIQE is treated as a descriptive no-reference metric.

---

# 3. Experimental Benchmarks and Evaluation

## 3.1 Held-out Evaluation

A held-out evaluation was conducted on 15 images using the final evaluation run. It produced 45 method-result rows across the three approaches, with no recorded evaluation failures.

### Held-out Results Ã¢â‚¬â€ 15 Images

| Method | PSNR Ã¢â€ â€˜ | SSIM Ã¢â€ â€˜ | LPIPS Ã¢â€ â€œ | NIQE Ã¢â€ â€œ | CIEDE2000 Ã¢â€ â€œ | Inference Latency (ms) |
|---|---:|---:|---:|---:|---:|---:|
| Classical CV | 16.3686 | 0.5048 | 0.3590 | 7.9804 | 16.2439 | 88.37 |
| Pretrained RetinexFormer | **21.7720** | **0.8224** | **0.1524** | **3.8765** | **8.9765** | 1960.97 |
| Hybrid Spatial-Adaptive | 18.1390 | 0.5885 | 0.2943 | 7.0345 | 13.7454 | 2395.17 |

*Arrows indicate the preferred direction of each metric. Latency is reported in milliseconds. Hybrid latency includes the sequential classical and deep-learning branches; fusion latency is measured separately.*

Standalone RetinexFormer achieved the best held-out PSNR, SSIM, LPIPS, NIQE and CIEDE2000 values among the three approaches.

Compared with standalone RetinexFormer, the hybrid method showed:
- PSNR difference: -3.6330 dB.
- SSIM difference: -0.2339.
- LPIPS difference: +0.1419.
- NIQE difference: +3.1580.
- CIEDE2000 difference: +4.7690.

The hybrid approach did not outperform standalone RetinexFormer on this held-out evaluation. This result informed the final production recommendation.

## 3.2 Validation and Model Selection

A fixed validation protocol was conducted using 97 paired images with validation seed 2026.

The predefined selection rule required positive paired-bootstrap 95% confidence interval lower bounds for PSNR, SSIM and improvement in negative LPIPS, plus mean NIQE and mean CIEDE2000 no worse than standalone RetinexFormer.

| Strategy | PSNR | SSIM | LPIPS | Mean Fusion Latency (ms) |
|---|---:|---:|---:|---:|
| Weighted 50:50 | 19.8884 | 0.6646 | 0.2922 | 11.73 |
| Adaptive Fusion | 18.8833 | 0.5983 | 0.3457 | 18.15 |
| Spatial-Adaptive Fusion | 18.4391 | 0.5711 | 0.3678 | 49.93 |
| DL-Dominant 90:10 | 21.8293 | 0.8286 | 0.1743 | 12.51 |
| DL-Dominant 95:5 | 21.9894 | 0.8399 | 0.1671 | 12.57 |
| DL-Dominant 80:20 | 21.4234 | 0.7932 | 0.1983 | 13.04 |
| Standalone RetinexFormer | **22.1058** | **0.8451** | **0.1645** | Ã¢â‚¬â€ |

No fusion candidate satisfied the complete predefined selection rule.

**Final selection:** Pretrained RetinexFormer.

## 3.3 Ablation Study

The validation compared fixed 50:50 weighted fusion, global adaptive fusion, spatial-adaptive fusion, and deep-learning-dominant 90:10, 95:5 and 80:20 fusion.

Increasing the deep-learning contribution improved performance relative to more balanced fusion configurations. However, none of the tested fusion strategies established reliable improvement over standalone RetinexFormer under the predefined statistical selection rule. Fusion is therefore retained as an experimental capability rather than selected as the production restoration method.

## 3.4 Evaluation Artifacts

Evaluation outputs are stored under `evaluation/results/`.

- `lolv2_our485_validation_seed2026.json`
- `lolv2_our485_fusion_validation_seed2026_verified.csv`
- `lolv2_our485_fusion_validation_seed2026_verified_summary.csv`
- `lolv2_our485_fusion_validation_seed2026_verified_decision.json`
- `lolv2_eval15_m3_final_seed2026.csv`
- `lolv2_eval15_m3_final_seed2026.csv.errors.txt`

The final held-out error log reports: `No evaluation failures.`

---

# 4. Failure Modes and Limitations

1. **Hybrid quality limitation:** The tested hybrid fusion method performed worse than standalone RetinexFormer on the held-out evaluation.
2. **Heuristic fusion:** The spatial contribution map is based on luminance heuristics and does not directly estimate restoration quality.
3. **Inference latency:** The deep-learning API has substantially higher inference latency than the classical branch in the reported CPU evaluation.
4. **Hardware dependency:** Deep-learning inference performance depends on available compute hardware.
5. **No-reference metric limitation:** NIQE is descriptive and should not be interpreted as a substitute for paired-reference restoration metrics.
6. **Generalization:** Broader testing across additional cameras, sensors and illumination conditions remains necessary.

The final production recommendation is standalone RetinexFormer based on the measured validation and held-out results.

---

# 5. Deployment and Quickstart

## 5.1 Project Repository

https://github.com/Vaishnavi-Dhanuka/CV-Hackathon-Project

## 5.2 Local Setup

Clone the repository with its submodules:

```bash
git clone --recurse-submodules https://github.com/Vaishnavi-Dhanuka/CV-Hackathon-Project.git
cd CV-Hackathon-Project
```

Create and activate a Python virtual environment, install the dependencies specified in the project setup guide, and ensure the required RetinexFormer checkpoint is available.

## 5.3 Start the Deep-Learning API

```bash
uvicorn deep_learning.api:app --host 127.0.0.1 --port 8000
```

## 5.4 Start the Streamlit Application

In another terminal:

```bash
streamlit run app/main.py
```

The Streamlit application communicates with the API using the configured `LUMINA_DL_API_URL` environment variable. Its default local API URL is `http://127.0.0.1:8000`.

## 5.5 Live Deployment

**Public application URL:** [ADD DEPLOYMENT URL WHEN AVAILABLE]

Confirm the deployment status and public URL before final submission.

---

# 6. Conclusion

LUMINA implements a modular low-light restoration application combining classical computer vision, pretrained deep learning and hybrid fusion.

The team evaluated the three approaches using paired-reference and no-reference quality metrics, latency measurements and a predefined validation-based selection procedure.

Although the hybrid pipeline demonstrates an adaptive mechanism for combining enhancement outputs, the measured experiments did not establish a reliable improvement over standalone RetinexFormer.

Accordingly, pretrained RetinexFormer is selected as the recommended production method, while the hybrid pipeline remains available for further experimentation.

The project provides a working application architecture, a separate inference API, comparative evaluation artifacts and documented limitations to support further development.
