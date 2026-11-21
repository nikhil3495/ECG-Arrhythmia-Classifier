# Architecture & engineering decision log

ADR-style record of every non-trivial decision taken while building this project (2026-11-21).
Numbers inside `<!-- BEGIN/END -->` blocks are generated from `results/*.json` by
`scripts/update_docs.py`; nothing quantitative here is typed by hand.

---

## ADR-001 - Build and run on Linux (WSL2 Ubuntu), Python 3.12

- **Context:** the dev machine is Windows 11. `litert-converter` (a hard dependency of
  `litert-torch`) only ships Linux/macOS wheels; there is no native `make`. The system Python in
  WSL is 3.14, which is too new for parts of the stack.
- **Decision:** develop and run everything inside WSL2 Ubuntu with a uv-managed CPython 3.12 venv.
  CI also uses Ubuntu + 3.12. The WSL VM is deliberately capped at 2 vCPUs/7 GB by the owner's
  `.wslconfig`; this was respected (not raised) - all training runs on 2 CPU threads.
- **Consequence:** `make all` is Linux/macOS only (documented in README).

## ADR-002 - Converter: `litert-torch` 0.10 (formerly `ai-edge-torch`), PT2E quantiser

- **Context:** the spec asked for Google's `ai-edge-torch`, possibly renamed. On PyPI the project is
  now published as **`litert-torch`** (`ai-edge-torch` 0.7.2 is the legacy name).
- **Decision:** use `litert_torch.convert` for FP32 and the **PT2E quantiser shipped with
  litert-torch** (`PT2EQuantizer` + `torchao` `prepare_pt2e/convert_pt2e`) for int8. It worked first
  time, so the onnx2tf fallback was not needed.
- **Verified:** the static int8 model is *full-integer*: int8 input tensors, int8 output tensor,
  and every compute tensor int8/int32 (inspected via the interpreter's tensor details). Its
  quantisation parameters are read from the flatbuffer at runtime (`ecg_edge/runtime.py`).
- **Extra row:** dynamic-range int8 (int8 weights, float activations) via the same quantiser with
  `is_dynamic=True`.

## ADR-003 - Inter-patient protocol as the primary result

- **Decision:** de Chazal et al. (2004) DS1/DS2 split - 22 training records, 22 test records, no
  patient in both. Paced records 102, 104, 107, 217 excluded (AAMI EC57). An intra-patient random
  80/10/10 beat split is reported **only as a secondary, explicitly leaky** comparison, trained for
  the selected architecture only.
- **Why:** random beat splits put neighbouring beats from the same person in train and test and
  inflate scores to ~99%; that does not measure generalisation to a new wearer.

## ADR-004 - Patient-wise validation set: records 101, 118, 203, 223 (from DS1)

- **Context:** early stopping and model selection need a validation set that never touches DS2.
- **Decision:** hold out 4 DS1 records chosen to contain S, V and F beats, while keeping the two
  records that hold most of DS1's minority classes - 209 (most S) and 208 (almost all F) - in
  training. Fixed in `configs/default.yaml` and never changed after seeing results.
- **Known weakness:** 203 (atrial flutter/fibrillation + multiform PVCs) and 118 (RBBB with APCs)
  are among the hardest records in the database, so validation scores are low and noisy. Only
  15 F beats remain in validation.

## ADR-005 - Model-selection signal: mean validation F1 over N, S, V

- **Context:** with 15 validation F beats, F-F1 on validation is noise (0 or near-0 for every run),
  and including it made early stopping erratic.
- **Decision:** early stopping and best-variant selection use the mean F1 over N, S and V on the
  validation records. The *reported* headline metric is still macro-F1 over **N, S, V, F** on DS2.

## ADR-006 - Headline macro-F1 over N, S, V, F (Q excluded from the average)

- **Context:** after removing paced records only 7 Q beats remain in DS2 (2 in train).
- **Decision:** the model still outputs all 5 AAMI classes, but the headline macro-F1 averages the
  four scored classes, as in de Chazal (2004) and most inter-patient literature. 5-class macro-F1
  is also stored in every results file (`macro_f1_5class`).

## ADR-007 - Class imbalance: inverse-frequency weighted cross-entropy

- **Options considered:** focal loss, balanced sampler, weighted CE.
- **Decision:** weighted cross-entropy with `w_c = (n_max / n_c)^p`, capped at 50. Simple,
  deterministic, no extra hyper-parameters beyond `p`, and does not repeat the 2 training Q beats
  thousands of times as a balanced sampler would.
- **Choice of p:** a validation-only sweep on the tiny variant (DS2 untouched) compared `p = 0.5`
  (sqrt-inverse) with `p = 1.0` over 3 seeds, plus regularisation variants:

<!-- BEGIN:sweep_table -->
| Run | Overrides vs. base | Best epoch | Val F1 (N,S,V) | Val F1 S | Val F1 V |
|---|---|---:|---:|---:|---:|
| w05 | train.class_weight_power=0.5 | 2 | 0.560 | 0.080 | 0.672 |
| w10 | train.class_weight_power=1.0 | 5 | 0.648 | 0.277 | 0.756 |
| w10_s1 | train.class_weight_power=1.0, seed=1 | 5 | 0.621 | 0.243 | 0.733 |
| w10_s2 | train.class_weight_power=1.0, seed=2 | 1 | 0.625 | 0.233 | 0.727 |
| w05_lr1e3_drop3_noise | train.lr=0.001, model.dropout=0.3, train.augment.noise_std=0.1 | 3 | 0.632 | 0.198 | 0.762 |
| w10_lr1e3 | train.class_weight_power=1.0, train.lr=0.001 | 4 | 0.607 | 0.162 | 0.768 |
| w10_reg | train.class_weight_power=1.0, model.dropout=0.3, train.augment.noise_std=0.1 | 5 | 0.606 | 0.189 | 0.711 |
| w10_reg_s1 | train.class_weight_power=1.0, model.dropout=0.3, train.augment.noise_std=0.1, seed=1 | 3 | 0.607 | 0.187 | 0.720 |
<!-- END:sweep_table -->

  All three `p = 1.0` seeds scored above the `p = 0.5` baseline, mostly through S-class F1, so
  it became the default. Heavier regularisation (dropout 0.3, noise 0.1) and a lower learning rate
  did not help. The sweep is small (8 runs, 1 variant); it guards against an obviously bad choice
  rather than claiming an optimum.

## ADR-008 - RR-interval branch with patient-normalised ratios

- **Decision:** 8 rhythm features per beat - pre-RR, post-RR, local mean RR (10 previous beats),
  long mean RR (previous 5 min), and the ratios pre/local, post/pre, pre/long, post/long. They are
  standardised *inside* the model (buffers), so deployed models consume raw features.
- **Why:** S beats are often morphologically normal; prematurity *relative to the patient's own
  rhythm* is the transferable cue. The two `/long` ratios were added after the first tiny run
  showed very low S recall on validation.
- **Causality note:** post-RR needs the next R-peak, i.e. one beat (~1 s) of latency in a streaming
  deployment. All averages use past beats only.

## ADR-009 - Architecture: Conv1d-BN-ReLU-MaxPool x4, GAP, concat RR MLP, FC head

- Only ops that lower to int8 LiteRT builtins (CONV_2D / DEPTHWISE_CONV_2D, MAX_POOL_2D, MEAN,
  FULLY_CONNECTED, CONCATENATION). BatchNorm is folded by the converter. No LSTM/attention/softmax
  in-graph (softmax is applied by the caller, so int8 logits keep full resolution).
- Three widths - tiny `[8,16,16,32]`, small `[16,32,32,64]`, base `[32,64,64,128]` - for the size/F1
  trade-off curve.

## ADR-010 - Training recipe

- AdamW (lr 2e-3, wd 1e-4), cosine schedule, batch 256, max 40 epochs, patience 8, seed 42.
- Light augmentation: amplitude x[0.85, 1.15], Gaussian noise sigma 0.05, circular shift +/-5 samples.
- CPU only (no GPU on the host); 2 threads. Per-variant timings are below.

<!-- BEGIN:training_table -->
| Variant | Params | Best epoch | Epochs run | Train time (s, CPU) | Val F1 (N,S,V) | Val macro-F1 (N,S,V,F) |
|---|---:|---:|---:|---:|---:|---:|
| **tiny** | 7,285 | 5 | 13 | 47 | 0.658 | 0.495 |
| small | 28,389 | 13 | 21 | 173 | 0.630 | 0.472 |
| base | 112,069 | 1 | 9 | 202 | 0.611 | 0.458 |
<!-- END:training_table -->

## ADR-011 - Calibration data for PTQ

- 2,000 beats sampled uniformly at random (seed 42) from the inter-patient **training** split
  only. Uniform sampling keeps the natural activation distribution the deployed model will see.

## ADR-012 - Latency methodology and labelling

- Batch 1, 1 thread, 100 warm-up + 1,000 timed runs, p50/p95 reported. For LiteRT the timed call is
  `interpreter.invoke()` with inputs pre-set, which matches what `benchmark_model` measures. For
  PyTorch it is an eager `model(beat, rr)` call under `inference_mode`.
- All numbers are **host x86 CPU** numbers and are labelled as such everywhere. Phone numbers
  require running `scripts/android_benchmark.md` on a device (not available tonight).

## ADR-013 - Deployment: GitHub Pages with in-browser LiteRT.js; Gradio app in Docker

- **Context:** `HF_TOKEN` was not set, so a Hugging Face Space could not be created.
- **Decision:** the live demo is a static GitHub Pages site (`docs/`) that runs the **same int8
  `.tflite` file** in the browser with LiteRT.js (`@litertjs/core`, WASM/XNNPACK backend), on bundled
  DS2 segments, and cross-checks every browser prediction against the Python LiteRT reference
  output stored next to it. The Gradio app (`app/app.py`, int8 model only) ships with a Dockerfile
  and is ready to push to a Space as soon as a token is available.

## ADR-014 - Model binaries live in GitHub Releases, not git

- `.tflite` / `.pt` files are git-ignored and attached to release v1.0.0. Exceptions: the single
  int8 model used by the demos (`app/models/ecg_int8.tflite`, `docs/demo/ecg_int8.tflite`), a few
  tens of KB each, are committed so that the Docker image and Pages site are self-contained.

## ADR-016 - No post-hoc re-selection after seeing test results

- **Observation:** validation selected **tiny** (training table above). On DS2, the base variant
  scores a higher macro-F1 than tiny (benchmark table in README), although base early-stopped
  at epoch 1 because the patient-wise validation signal is very noisy.
- **Decision:** keep the validation-selected model as the headline. Swapping to base, or
  retuning base's early stopping, *after* seeing DS2 scores would turn the test set into a
  validation set and inflate the reported number. The full curve is published so readers can
  see all three widths, and the noisy-selection problem is documented as a limitation.
- **Future fix (not applied tonight):** k-fold *patient-wise* cross-validation over DS1 records,
  or checkpoint averaging, for a less noisy selection signal.

## ADR-017 - Fold RR standardisation into the first Linear layer at export

- **Context:** the first int8 export had int8 beat input and int8 output, but the in-graph RR
  standardisation `(rr - mu) / sigma` was not annotated by the PT2E quantiser. It stayed as float
  `SUB`/`MUL` ops on a float32 RR input, followed by a `QUANTIZE` op, so the model was not
  integer-only. The LiteRT.js browser test exposed this (it rejected an int8 RR tensor).
- **Decision:** `fold_rr_normalization()` rewrites `W((rr - mu)/sigma) + b` as
  `(W/sigma) rr + (b - W mu/sigma)` on a copy of the model before conversion. It is exact (unit test
  `test_fold_rr_normalization_is_exact`); FP32 `.tflite` parity with the unfolded PyTorch model
  stays at 100%. The exported int8 models now contain only int8/int32 tensors, and both inputs
  are int8 (asserted in `test_int8_tflite_is_full_integer_and_close`).
- This was adopted for structural reasons (the integer-only requirement) and kept regardless of its
  effect on scores. The pre-fold run was not kept as a result file, so it is not quoted here.

## ADR-015 - Filtering is zero-phase (offline)

- `sosfiltfilt` is non-causal. It is fine for this offline benchmark and the demo, but a
  streaming on-device pipeline should switch to a causal filter (listed as future work).
