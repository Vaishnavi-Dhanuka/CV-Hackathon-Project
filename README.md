# Robust Low-Light Image Restoration and Enhancement

A deep-learning-based low-light image enhancement system built around **Retinexformer**, with classical image-processing comparisons and fine-tuned model variants.

## 1. Project Overview

Low-light images often suffer from poor visibility, noise, colour distortion, and loss of detail. This project evaluates deep-learning approaches for restoring image brightness and visual quality while preserving structural information.

**Primary architecture:** Retinexformer
**Framework:** PyTorch
**Image processing:** OpenCV, NumPy
**Evaluation:** PSNR, SSIM, LPIPS, inference latency

## 2. Implemented Work

- Integrated the official pretrained Retinexformer architecture.
- Prepared LOLv2 Real Captured training and validation data.
- Prepared a second training configuration combining LOLv2 Real and Synthetic data.
- Fine-tuned two Retinexformer variants:
  - **Model A:** LOLv2 Real Captured.
  - **Model B:** LOLv2 Real Captured + LOLv2 Synthetic.
- Compared Retinexformer with Multinex and Zero-DCE++.
- Evaluated classical enhancement and hybrid fusion approaches.
- Implemented image inference, batch testing, and quantitative evaluation scripts.

## 3. Dataset

The experiments use:

- **LOLv2 Real Captured:** paired low-light and normal-light images.
- **LOLv2 Synthetic:** paired synthetic low-light and normal-light images.
- **ExDark:** unpaired low-light images intended for external qualitative evaluation.

The datasets are not included in this repository. Download them separately and prepare the directory structure expected by the configuration files and dataset scripts.

## 4. Model Results

### Architecture comparison

Results on 69 LOLv2 Real validation images.

| Model | PSNR â†‘ | SSIM â†‘ | LPIPS â†“ | Latency (ms) â†“ |
|---|---:|---:|---:|---:|
| Retinexformer | 21.4968 | 0.7896 | 0.2372 | 88.52 |
| Multinex | 19.2540 | 0.7804 | 0.2538 | 227.13 |
| Zero-DCE++ | 13.3230 | 0.3933 | 0.6117 | 7.63 |

Retinexformer achieved the strongest image-quality results among the evaluated architectures. Zero-DCE++ had the lowest latency but substantially lower restoration quality.

### Fine-tuning comparison

Model B validation results:

| Dataset | Model | PSNR â†‘ | SSIM â†‘ | LPIPS â†“ |
|---|---|---:|---:|---:|
| Real | Baseline | 21.4968 | 0.7896 | 0.2368 |
| Real | Model B | 22.0533 | 0.7936 | 0.2316 |
| Synthetic | Baseline | 17.0300 | 0.6961 | 0.2870 |
| Synthetic | Model B | 17.8469 | 0.7226 | 0.2741 |

### Official LOLv2 Real test comparison

Results on the 100-image official test set:

| Model | PSNR â†‘ | SSIM â†‘ | LPIPS â†“ |
|---|---:|---:|---:|
| Pretrained baseline | 22.7897 | 0.8384 | 0.1708 |
| Model A | 22.5491 | 0.8387 | 0.1725 |
| Model B | 22.5928 | 0.8399 | 0.1697 |

**Model selection:** The original pretrained Retinexformer is the primary model based on its strongest official-test PSNR. Model B is retained as an alternative, with better official-test SSIM and LPIPS and improved validation results on both Real and Synthetic data.

These results apply to the evaluated datasets and configurations; they do not establish universal superiority.

## 5. Repository Structure

```text
CV_Project/
â”œâ”€â”€ configs/
â”‚   â””â”€â”€ retinexformer/
â”‚       â”œâ”€â”€ RetinexFormer_Model_A.yml
â”‚       â”œâ”€â”€ RetinexFormer_Model_B.yml
â”‚       â””â”€â”€ RetinexFormer_finetune.yml
â”œâ”€â”€ deep_learning/
â”‚   â”œâ”€â”€ api.py
â”‚   â”œâ”€â”€ retinex_inference.py
â”‚   â”œâ”€â”€ test_inference.py
â”‚   â”œâ”€â”€ evaluate_retinexformer.py
â”‚   â”œâ”€â”€ evaluate_validation.py
â”‚   â”œâ”€â”€ compare_model_A.py
â”‚   â”œâ”€â”€ compare_model_B.py
â”‚   â”œâ”€â”€ evaluate_model_A_test.py
â”‚   â”œâ”€â”€ evaluate_model_B_test.py
â”‚   â”œâ”€â”€ prepare_finetune_data.py
â”‚   â”œâ”€â”€ prepare_model_datasets.py
â”‚   â””â”€â”€ rebuild_model_B.py
â”œâ”€â”€ third_party/
â”‚   â””â”€â”€ Retinexformer/  # Git submodule
â”œâ”€â”€ results/             # Local; not committed
â”œâ”€â”€ data/                # Local; not committed
â”œâ”€â”€ .gitignore
â”œâ”€â”€ .gitmodules
â””â”€â”€ README.md
```

## 6. Setup

Requirements: Python 3.11 and a compatible PyTorch installation.

Clone the project and initialize the Retinexformer submodule:

```bash
git clone --recurse-submodules https://github.com/Vaishnavi-Dhanuka/CV-Hackathon-Project.git
cd CV-Hackathon-Project
```

Create and activate a virtual environment, then install the required dependencies for the project and Retinexformer.

Download the official Retinexformer LOLv2 Real pretrained checkpoint and place it at:

```text
third_party/Retinexformer/pretrained_weights/LOL_v2_real.pth
```

The pretrained checkpoint and fine-tuned checkpoints are not stored in GitHub.

The primary M3 inference API uses the verified official baseline at
`results/checkpoints/official_only/retinexformer_official_baseline.pth`
(SHA-256:
`539bd16c4da6179e45616329f249c4672951b1045193428e1d042c50d4b65a0b`).
For local use, place that checkpoint at the stated path. Render downloads and
verifies it at startup through `scripts/fetch_render_checkpoint.py`; do not
commit the checkpoint. The original M2 evaluation script instead reads
`third_party/Retinexformer/pretrained_weights/LOL_v2_real.pth`.

The LOL-v2 Real Captured test dataset ZIP is supplied separately as an external
handoff and is not tracked in Git. Keep the ZIP outside the repository and
extract its `Test/Low` and `Test/Normal` directories under
`data/raw/Kaggle/archive/LOL-v2/Real_captured/` to run
`deep_learning/evaluate_retinexformer.py`. Model A/B checkpoints and training
datasets are optional and are not needed for the primary inference pipeline.

The M3 FastAPI production API, Render service configuration, and Streamlit
backend URL configuration remain in place. The deployed frontend receives
`LUMINA_DL_API_URL` from the backend service URL; see
[`RENDER_DEPLOYMENT.md`](./RENDER_DEPLOYMENT.md) for deployment and verification
details.

## 7. Inference

The main inference implementation is:

```text
deep_learning/retinex_inference.py
```

The batch evaluation script is:

```text
deep_learning/test_inference.py
```

The API implementation is available in:

```text
deep_learning/api.py
```

See the scripts for their required arguments and local checkpoint paths.

## 8. Reproducing Fine-Tuning

Prepare the dataset splits using:

```bash
python deep_learning/prepare_finetune_data.py
```

Prepare or rebuild Model B data using:

```bash
python deep_learning/rebuild_model_B.py
```

Fine-tuning configurations are in:

```text
configs/retinexformer/
```

The training configurations expect the local dataset and pretrained checkpoint paths specified in their YAML files. The Retinexformer training implementation is provided by the submodule.

## 9. Evaluation

The repository includes scripts for:

- Baseline and fine-tuned model evaluation.
- Model A and Model B comparisons.
- Official test-set comparisons.
- Architecture benchmarking.
- Hybrid fusion and detail-refinement experiments.

Evaluation CSVs and generated images are retained locally and are not all committed to the repository.

## 10. Team

Collaborative project developed by a team of four members for the computer vision hackathon.
