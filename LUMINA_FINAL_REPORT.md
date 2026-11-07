# M3 Integration and Evaluation Completion

## Outcome

The M3 frontend, API clients, evaluation runner, validation experiment, and
held-out report are in the worktree. The official RetinexFormer checkpoint
remains the production DL model. Seeded validation selected RetinexFormer
alone; the current spatial Hybrid implementation remains experimental because
its measured quality was lower on both validation and eval15.

## Production model and deployment

- **Architecture:** official RetinexFormer implementation,
  `deep_learning/retinex_inference.py`, copied from the requested
  `origin/deep-learning` version and used by the existing API.
- **Checkpoint:** `results/checkpoints/official_only/retinexformer_official_baseline.pth`
- **Checkpoint verification:** 6,478,393 bytes; SHA-256
  `539bd16c4da6179e45616329f249c4672951b1045193428e1d042c50d4b65a0b`.
- **Loading:** strict load from the checkpoint's `params` state dictionary.
  The model is cached, and CPU is selected when CUDA is unavailable. Runtime
  verification reported `torch.cuda.is_available() == False`.
- **API smoke test:** `POST /enhance` returned HTTP 200 and a PNG RGB image at
  the original 16 x 16 size.
- **Submodule:** `third_party/Retinexformer/` was present at
  `1e9a0efce4b306b6701b824768370ff26066c32a`.

Start the services from the repository root in separate PowerShell windows:

```powershell
python -m uvicorn deep_learning.api:app --host 127.0.0.1 --port 8000
python -m streamlit run app/main.py
```

## M3 implementation delivered

- Model-agnostic DL and Classical CV adapters under `hybrid/`, retaining RGB
  `uint8` outputs, original dimensions, latency measurements, and clear errors.
- LUMINA modes for Original Image, Classical CV, Official RetinexFormer, and
  Hybrid Intelligence; image-content/mode caching, retries, downloads,
  loading feedback, metrics, and heuristic fusion-contribution visualizations.
- Paired evaluation runner, result visualizer, fixed-split fusion experiment,
  and non-overwriting eval15 finalizer.
- Setup/architecture documentation and the evaluation guide and reports.

The following are M3-owned files delivered or updated in this worktree:

- `app/main.py`
- `app/components/sidebar.py`
- `app/components/image_display.py`
- `app/components/metrics_display.py`
- `app/components/fusion_explainability.py`
- `app/tests/test_pipeline_modes.py`
- `app/tests/test_dl_integration.py`
- `app/tests/test_hybrid_integration.py`
- `app/tests/test_fusion_explainability.py`
- `hybrid/dl_client.py`
- `hybrid/classical_client.py`
- `hybrid/__init__.py`
- `hybrid/fusion.py`
- `hybrid/tests/test_dl_client.py`
- `hybrid/tests/test_classical_client.py`
- `hybrid/tests/test_fusion.py`
- `hybrid/tests/test_official_dl_inference.py`
- `evaluation/runner.py`
- `evaluation/visualize_results.py`
- `evaluation/fusion_experiments.py`
- `evaluation/finalize_benchmark.py`
- `evaluation/README.md`
- `evaluation/results_template.csv`
- `evaluation/tests/test_runner.py`
- `evaluation/tests/test_visualize_results.py`
- `evaluation/tests/test_fusion_experiments.py`
- `evaluation/tests/test_finalize_benchmark.py`
- `requirements.txt`
- `LUMINA_SETUP_GUIDE.md`
- `LUMINA_ARCHITECTURE.md`
- `LUMINA_EVALUATION_REPORT.md`
- `LUMINA_FINAL_REPORT.md`

The generated validation and eval15 artifacts are in `evaluation/results/`.
The fixed validation split manifest is
`evaluation/results/lolv2_our485_validation_seed2026.json`.

The worktree also contained pre-existing staged submodule registration
(`.gitmodules` and the `third_party/Retinexformer` gitlink) and a local
`third_party/Retinexformer_M3_backup/`; these were preserved. The listed
fusion-related files were already part of the M3 worktree and were not edited
to perform validation or change fusion behavior in this completion.

## Evaluation decision

The fixed seed-2026 80/20 split used 97 validation pairs from original LOL (LOL-v1) `our485`.
No fusion candidate passed the predeclared paired-bootstrap criteria, so the
recommended production result is RetinexFormer alone. After that decision was
fixed, the one-shot eval15 run processed all 15 pairs (45 method rows) without
failures. Its measured averages were:

| Method | PSNR â†‘ | SSIM â†‘ | LPIPS â†“ | NIQE â†“ | Inference ms | Fusion ms |
|---|---:|---:|---:|---:|---:|---:|
| Classical CV | 16.3686 | 0.5048 | 0.3590 | 7.9804 | 88.37 | N/A |
| Pretrained Retinexformer | 21.7720 | 0.8224 | 0.1524 | 3.8765 | 1960.97 | N/A |
| Hybrid Intelligence | 18.1390 | 0.5885 | 0.2943 | 7.0345 | 2395.17 | 47.51 |

These are measurements from this run only. NIQE is no-reference, latency is
not accuracy, and the results do not support claiming that Hybrid improves
restoration quality.

See [LUMINA_EVALUATION_REPORT.md](./LUMINA_EVALUATION_REPORT.md) for all validation
candidate means, eval15 color-fidelity results, relative comparisons, and
artifact paths. The pre-existing official eval15 CSV remains unchanged.

## Validation performed

- `python -m pytest evaluation/tests hybrid/tests app/tests -v` â€” **129 passed,
  13 warnings, 7 subtests passed**.
- `git diff --check` â€” passed.
- Streamlit AppTest smoke test â€” rendered the four pipeline modes without
  exceptions.
- API smoke test â€” HTTP 200, PNG/RGB, original dimensions.
- Seeded our485 validation â€” 97/97 images completed; no failures.
- One-shot eval15 finalizer â€” 15/15 images and 45/45 method rows completed;
  no failures.

The warnings are dependency deprecations (Matplotlib/PyParsing and
torchvision/LPIPS-related); they did not fail tests. The local Python
environment also reports unrelated `pip check` conflicts involving
`sktime`/NumPy/scikit-learn and `torchaudio`/PyTorch.

## Preserved / left untouched

- M1 `classical/` implementation and M2 API/experimental model source were not
  edited. The official inference file above was brought in from the specified
  upstream branch as requested; it was not reimplemented.
- The fusion calculations in `hybrid/fusion.py` were not changed for this
  evaluation. Hybrid contribution maps are heuristic weights, not learned
  attention or evidence of improved restoration.
- The official checkpoint and LOLv2 data/ground truth were not changed.
- The existing `evaluation/results/lolv2_eval15_official_checkpoint.csv`,
  submodule registration, pinned RetinexFormer submodule, and local
  `third_party/Retinexformer_M3_backup/` were left intact.
- No commit or push was made.

## Remaining limitations

- The existing spatial-adaptive fusion is not selected for production because
  it underperformed RetinexFormer on the measured validation and eval15
  comparisons. It remains available for transparent experimental comparison.
- CPU API round-trip latency averaged about 1.96 seconds for standalone
  RetinexFormer on eval15 in this environment; timings vary with hardware and
  server load.
- Reproducing model inference requires the separately supplied official
  checkpoint and initialized RetinexFormer submodule. The setup steps are in
  [LUMINA_SETUP_GUIDE.md](./LUMINA_SETUP_GUIDE.md).

