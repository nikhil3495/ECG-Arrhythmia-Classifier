# Measuring true on-device latency on an Android phone

All latency numbers currently in this repository were measured on a **desktop x86-64 CPU**
(see `results/benchmark.json -> hardware`). They are **not** phone numbers. This guide shows how to
measure real on-device latency with the official LiteRT `benchmark_model` tool over `adb`.

## 1. Prerequisites

- An Android phone (arm64-v8a) with **Developer options -> USB debugging** enabled.
- Android platform tools (`adb`) on your PATH: `adb devices` should list the phone.
- The models: either run `make export` (they land in `exports/`) or download them from the
  [v1.0.0 release](https://github.com/nikhil3495/ECG-Arrhythmia-Classifier/releases/tag/v1.0.0).

## 2. Get the prebuilt `benchmark_model` binary

LiteRT publishes nightly prebuilt benchmark binaries (see the LiteRT docs, "Performance measurement"):

```bash
# arm64 Android, plain CPU (XNNPACK) build
curl -L -o benchmark_model \
  https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_benchmark_model
chmod +x benchmark_model
```

If that URL moves, build it from source:
`bazel build -c opt --config=android_arm64 //tflite/tools/benchmark:benchmark_model` in the
[google-ai-edge/LiteRT](https://github.com/google-ai-edge/LiteRT) repo.

## 3. Push binary and models

```bash
adb push benchmark_model /data/local/tmp/
adb shell chmod +x /data/local/tmp/benchmark_model
for v in tiny small base; do
  for q in fp32 int8 int8_dynamic; do
    adb push exports/${v}_${q}.tflite /data/local/tmp/
  done
done
```

## 4. Run (single thread, batch 1, same protocol as the desktop numbers)

```bash
for v in tiny small base; do
  for q in fp32 int8 int8_dynamic; do
    echo "== ${v}_${q}"
    adb shell /data/local/tmp/benchmark_model \
      --graph=/data/local/tmp/${v}_${q}.tflite \
      --num_threads=1 --use_xnnpack=true \
      --warmup_runs=100 --num_runs=1000 \
      --enable_op_profiling=false 2>&1 | grep -E "Inference timings|Peak memory"
  done
done
```

Notes for clean numbers:

- Keep the phone plugged in, screen on, and idle; close other apps. Run each model twice and keep
  the second run (thermal/frequency warm-up).
- To pin to a big or little core, prefix with `taskset <mask>` (e.g. `taskset f0` for cores 4-7).
- `--enable_op_profiling=true` prints per-op timings (useful to see whether int8 convs hit XNNPACK).
- `benchmark_model` reports `avg`, `std`, `min`, `max` in microseconds. For p50/p95 use
  `--report_peak_memory_footprint=true --output_prefix=...` or log individual runs with
  `--print_postinvoke_state=false --run_delay=0` and post-process.

## 5. Results table - **to be filled after running on device**

> Device: _<phone model>_ - SoC: _<SoC>_ - Android _<version>_ - core: _<big/little>_ - date: _<date>_

| Variant | Runtime | Size (KB) | avg latency (us) | std (us) | Peak memory (MB) |
|---|---|---:|---:|---:|---:|
| tiny  | LiteRT FP32 | | _to be filled after running on device_ | | |
| tiny  | LiteRT int8 | | _to be filled after running on device_ | | |
| small | LiteRT FP32 | | _to be filled after running on device_ | | |
| small | LiteRT int8 | | _to be filled after running on device_ | | |
| base  | LiteRT FP32 | | _to be filled after running on device_ | | |
| base  | LiteRT int8 | | _to be filled after running on device_ | | |

Accuracy/F1 do not need re-measuring on the phone: the same `.tflite` flatbuffers were already run
with the LiteRT interpreter on the full DS2 test set (`results/export.json`); only latency is
hardware-dependent.
