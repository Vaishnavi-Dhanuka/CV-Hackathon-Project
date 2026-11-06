
# Intelligent Low-Light Image Restoration and Enhancement

An illumination-adaptive computer vision system that restores severely underexposed, noisy, and colour-distorted images while preserving fine textures, edges, and structural information.

## Problem Statement

**P1 – Robust Low-Light Image Restoration and Enhancement**

Conventional brightness enhancement techniques often introduce overexposure, amplify sensor noise, distort colours, and lose important image details.

Our goal is to develop an intelligent restoration system that adapts to different illumination conditions and produces visually natural, high-quality images.

## Project Objectives

- Estimate illumination and identify low-light regions.
- Enhance brightness and contrast adaptively.
- Suppress noise without destroying fine details.
- Restore natural colours and improve colour fidelity.
- Preserve edges, textures, and structural information.
- Combine classical computer vision and deep learning.
- Evaluate performance using objective image-quality metrics.

## Proposed Solution

We propose a hybrid low-light image enhancement framework combining classical image processing techniques with deep learning.

### 1. Classical Computer Vision
- Retinex-based illumination estimation and decomposition.
- CLAHE for adaptive contrast enhancement.
- Bilateral or guided filtering for edge-preserving denoising.
- Gamma correction and colour correction.
- Adaptive parameter selection based on image characteristics.

### 2. Deep Learning
- Implement or adapt a low-light enhancement model.
- Explore Zero-DCE / Zero-DCE++ and RetinexNet.
- Fine-tune models using suitable low-light datasets.
- Evaluate generalization across different lighting conditions.

### 3. Hybrid Intelligence
- Combine illumination estimation with deep learning restoration.
- Adaptively select or blend enhancement outputs.
- Preserve textures and edges while suppressing noise.
- Avoid overexposure, halo artifacts, and unnatural colours.

## Proposed Pipeline

Input Image  
→ Image Analysis and Illumination Estimation  
→ Adaptive Enhancement Strategy  
→ Classical CV / Deep Learning Enhancement  
→ Hybrid Fusion  
→ Colour and Detail Refinement  
→ Quality Evaluation  
→ Enhanced Output

## Comparative Approaches

1. Classical Computer Vision Only
2. Deep Learning Only
3. Hybrid Classical + Deep Learning

## Evaluation Metrics

- PSNR – Peak Signal-to-Noise Ratio
- SSIM – Structural Similarity Index
- LPIPS – Learned Perceptual Image Patch Similarity
- NIQE – Natural Image Quality Evaluator
- Processing Latency
- Edge Preservation and Noise Suppression

## Ablation Study

We will evaluate the contribution of individual components by comparing:

- Without illumination estimation.
- Without denoising.
- Without colour correction.
- Without adaptive fusion.
- Complete hybrid pipeline.

## Technology Stack

| Component | Technology |
|---|---|
| Programming | Python |
| Image Processing | OpenCV, NumPy |
| Deep Learning | PyTorch |
| Enhancement Models | Zero-DCE++, RetinexNet |
| Evaluation | scikit-image, LPIPS, NIQE |
| Interface | Streamlit |
| Version Control | Git and GitHub |

## Repository Structure

```text
CV-Hackathon-Project/
│
├── data/
│   ├── low_light/
│   └── reference/
│
├── classical/
│   ├── illumination.py
│   ├── retinex.py
│   ├── clahe.py
│   ├── denoising.py
│   └── color_correction.py
│
├── deep_learning/
│   ├── models/
│   ├── train.py
│   └── inference.py
│
├── hybrid/
│   └── adaptive_fusion.py
│
├── evaluation/
│   ├── metrics.py
│   └── ablation.py
│
├── app/
│   └── main.py
│
├── results/
├── notebooks/
├── requirements.txt
└── README.md
```

## Expected Deliverables

- Adaptive low-light image restoration system.
- Classical, deep learning, and hybrid implementations.
- Comparative performance evaluation.
- Ablation study and failure analysis.
- Intermediate enhancement visualizations.
- Interactive deployed application.

## Team

Collaborative development by a team of three members.

## Project Status

🚧 Development in progress.
