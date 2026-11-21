# Handoff - overnight build (2026-11-21)

## Links

| What | URL |
|---|---|
| Repository | https://github.com/nikhil3495/ECG-Arrhythmia-Classifier |
| Live demo (GitHub Pages, int8 LiteRT.js in-browser) | https://nikhil3495.github.io/ECG-Arrhythmia-Classifier/ |
| Release v1.0.0 (models + checkpoint) | https://github.com/nikhil3495/ECG-Arrhythmia-Classifier/releases/tag/v1.0.0 |
| CI | https://github.com/nikhil3495/ECG-Arrhythmia-Classifier/actions/workflows/ci.yml |

## Final results (inter-patient DS2, best model chosen on validation)

<!-- BEGIN:headline -->
Best model (**tiny**, chosen on validation): full-integer **int8 LiteRT, 21.7 KB**, **macro-F1 0.585** (N,S,V,F) and accuracy 0.891 on 49,668 beats from 22 *unseen* patients (DS2); **2.1x smaller** than its FP32 .tflite with a macro-F1 change of -0.0013; p50 single-beat latency **0.008 ms** (Desktop x86_64 CPU (12th Gen Intel(R) Core(TM) i5-12500H), 1 thread, batch 1 - NOT a phone).
<!-- END:headline -->

<!-- BEGIN:headline_table -->
| Model | Size (KB) | Accuracy | Macro-F1 (N,S,V,F) | F1 N | F1 S | F1 V | F1 F | p50 latency (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tiny - PyTorch FP32 | 39.4 | 0.882 | 0.586 | 0.937 | 0.517 | 0.875 | 0.017 | 0.120 |
| tiny - LiteRT FP32 | 45.8 | 0.882 | 0.586 | 0.937 | 0.517 | 0.875 | 0.017 | 0.013 |
| tiny - LiteRT int8 (full-integer) | 21.7 | 0.891 | 0.585 | 0.942 | 0.513 | 0.872 | 0.012 | 0.008 |
<!-- END:headline_table -->

<!-- BEGIN:per_class_table -->
| Class | Support | Precision (PPV) | Recall (Se) | F1 |
|---|---:|---:|---:|---:|
| N | 44,218 | 0.987 | 0.901 | 0.942 |
| S | 1,836 | 0.372 | 0.827 | 0.513 |
| V | 3,219 | 0.861 | 0.884 | 0.872 |
| F | 388 | 0.007 | 0.036 | 0.012 |
| Q | 7 | 0.000 | 0.000 | 0.000 |
<!-- END:per_class_table -->

<!-- BEGIN:quant_table -->
| Variant | FP32 -> int8 size | Macro-F1 delta | Accuracy delta | Speed-up vs LiteRT FP32 | Speed-up vs PyTorch | FP32 .tflite agreement w/ PyTorch | int8 agreement w/ PyTorch |
|---|---:|---:|---:|---:|---:|---:|---:|
| tiny | 2.11x smaller | -0.0013 | +0.0091 | 1.54x | 14.2x | 100.00% | 98.32% |
| small | 2.87x smaller | +0.0003 | +0.0058 | 2.40x | 10.0x | 100.00% | 98.54% |
| base | 3.47x smaller | -0.0002 | -0.0053 | 3.18x | 6.6x | 100.00% | 99.12% |
<!-- END:quant_table -->

## What was verified

- `make test` (pytest, synthetic fixtures), `ruff check`, `black --check`: all pass locally and in CI.
- FP32 `.tflite` vs PyTorch parity on all DS2 beats, for every width (see `results/export.json`).
- The int8 models are integer-only (int8 inputs and output, only int8/int32 tensors); asserted by tests.
- In-browser LiteRT.js predictions are identical to Python LiteRT on every bundled beat tested
  (headless Chrome); the Pages URL returns HTTP 200.
- The Gradio app (`make app`) and the Docker image both start and return HTTP 200 locally.
- A fresh `git clone` into /tmp followed by `make setup`, `make test` (32 passed) and `make lint` succeeded;
  CI separately installs `requirements.txt` with plain pip on a clean runner.

## Needs your action

1. **Android on-device latency.** All latency numbers are desktop x86 CPU (i5-12500H, WSL2,
   1 thread). Follow `scripts/android_benchmark.md` with a phone over adb and fill in its table.
2. **Hugging Face Space (optional).** `HF_TOKEN` was not set, so no Space was created. To add one:
   `export HF_TOKEN=...`, create a Gradio Space `nikhil3495/ecg-arrhythmia-classifier`, and upload
   `app/` + `src/ecg_edge/{__init__,runtime}.py` (the Dockerfile shows the exact file set). The
   Pages demo already covers the "live demo" requirement.
3. **Model selection caveat to know before interviews.** Validation picked *tiny*; *base* scores
   higher on DS2. This is documented (ADR-016) rather than "fixed" on test data. If you want a
   stronger headline, run patient-wise k-fold model selection on DS1 (future work).
4. The repo's git author is `nikhil3495@users.noreply.github.com` (WSL git config). Change it if you
   want your personal email on the commits.

## Environment notes

- Built in WSL2 Ubuntu (`~/ECG-Arrhythmia-Classifier`), Python 3.12 via uv, CPU only, 2 threads
  (your `.wslconfig` caps WSL at 2 CPUs; left unchanged).
- `litert-torch` (the new name of `ai-edge-torch`) has no Windows wheels; use Linux/macOS/WSL.
- Every decision is logged in `docs/DECISIONS.md` (ADR-001 to ADR-017).
