# Failure Modes, Limitations, and Responsible Reporting

## 1. Observed evaluation weakness: hybrid fusion

The M3 spatial-adaptive Hybrid Intelligence result underperformed standalone pretrained RetinexFormer on the reported 97-image validation and 15-image eval15 comparisons. On eval15, Hybrid was lower by 3.6330 dB PSNR and 0.2339 SSIM, and higher by 0.1419 LPIPS, 3.1580 NIQE, and 4.7690 CIEDE2000.

**Interpretation:** the current heuristic blend can dilute the stronger RetinexFormer output. The contribution maps show blend weights, not per-pixel correctness or quality.

**Mitigation:** retain standalone RetinexFormer as the production recommendation; keep hybrid as an experimental comparison. Any proposed fusion improvement must be evaluated on validation data and then tested on a previously untouched hold-out.

## 2. Classical CV limitations

The classical branch depends on M1 preprocessing and guided Retinex. Its output can differ from the learned model in brightness, contrast, and colour. The eval15 mean metrics are lower than RetinexFormer in the reported run.

**Mitigation:** keep it as a baseline and inspect representative failure examples. Do not present it as the production quality winner based on current results.

## 3. Low-light restoration generalization

The available benchmark results are dataset- and protocol-specific. They do not establish performance on every camera sensor, lighting condition, noise distribution, or unseen deployment domain.

**Mitigation:** include genuinely unseen images in the demo; record the source and conditions; avoid tuning on the final held-out set; document failures as well as successful examples.

## 4. Colour and perceptual trade-offs

Fusion of two different outputs can alter colour or contrast. CIEDE2000 provides a colour-difference diagnostic but does not replace PSNR, SSIM, or LPIPS. NIQE is no-reference and is not a direct comparison with paired ground truth.

**Mitigation:** report metrics separately with their direction and interpretation. Include visual examples where metric and human perception may disagree.

## 5. Runtime and deployment risks

The M3 report records approximately 1.96 seconds average standalone RetinexFormer inference latency on eval15 in the local CPU environment. Hardware and server load affect timing. Local smoke tests do not establish Render cold-start, memory, or public-service reliability.

**Mitigation:** test the actual hosted URL from a fresh browser/device, measure first-request and subsequent-request latency, test representative image sizes, and record failures. The Render Blueprint currently declares Free plans; suitability must be confirmed empirically.

## 6. Input/API failure risks

The API has invalid-image handling, upload-size and decoded-pixel limits, missing-checkpoint handling, and production-safe generic errors. The client expects a PNG response and validates dimensions. Network failures and timeouts can still interrupt the user workflow.

**Mitigation:** test invalid files, oversized files, unavailable backend, missing checkpoint, timeout, and malformed responses. Ensure the UI presents a clear actionable error and does not silently treat a failed request as a valid result.

## 7. Reproducibility risks

A fresh clone does not include the production checkpoint or datasets. A successful local run may rely on local files that are absent from Git.

**Mitigation:** document authorized checkpoint acquisition, verify the checkpoint SHA-256, pin the submodule, record package versions, provide exact dataset placement instructions, and preserve per-image result artifacts.

## 8. Ethics and responsible presentation

Enhancement can change the appearance of an image and may create or suppress visual details. Enhanced output should not be represented as a faithful recovery of information that was never captured. Avoid using enhanced imagery as sole evidence in safety-critical, forensic, medical, or legal contexts.


## 9. Image-level failure evidence

The reviewed project summaries do not identify specific named image-level failure cases or their exact file paths. Therefore, this report limits its failure analysis to the documented aggregate evaluation results and known system limitations. It does not claim image-specific failure examples or unverified causes.
