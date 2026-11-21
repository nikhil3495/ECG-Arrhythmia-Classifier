"""Size / accuracy / F1 / single-beat latency benchmark for every variant x runtime.

Latency is measured on the machine running this script (recorded in ``hardware``), batch 1,
single thread, after warm-up. It is *host CPU* latency, not phone latency - see
``scripts/android_benchmark.md`` for on-device measurement with LiteRT's ``benchmark_model``.
"""

from __future__ import annotations

import argparse
import io
import os
import platform
import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch

from ecg_edge import AAMI_CLASSES
from ecg_edge.data import inter_patient_splits, load_dataset
from ecg_edge.models import count_params, load_checkpoint
from ecg_edge.runtime import LiteRTClassifier
from ecg_edge.utils import CHECKPOINT_DIR, EXPORT_DIR, RESULTS_DIR, get_logger, load_config, load_json, save_json

log = get_logger(__name__)
VARIANTS = ("tiny", "small", "base")
RUNTIMES = ("pytorch_fp32", "litert_fp32", "litert_int8", "litert_int8_dynamic")
RUNTIME_LABELS = {
    "pytorch_fp32": "PyTorch FP32",
    "litert_fp32": "LiteRT FP32",
    "litert_int8": "LiteRT int8 (full-integer)",
    "litert_int8_dynamic": "LiteRT int8 (dynamic-range)",
}


def cpu_name() -> str:
    """Best-effort CPU model string."""
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def hardware_info(num_threads: int) -> dict[str, Any]:
    """Describe where latency was measured."""
    release = platform.release()
    return {
        "cpu": cpu_name(),
        "arch": platform.machine(),
        "os": f"{platform.system()} {release}",
        "wsl2": "microsoft" in release.lower(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "threads": num_threads,
        "label": f"Desktop {platform.machine()} CPU ({cpu_name()}), {num_threads} thread, batch 1 - NOT a phone",
    }


def time_fn(fn: Callable[[], Any], warmup: int, runs: int) -> dict[str, float]:
    """Latency percentiles (ms) of ``fn`` over ``runs`` calls after ``warmup`` calls."""
    for _ in range(warmup):
        fn()
    times = np.empty(runs)
    for i in range(runs):
        t0 = time.perf_counter_ns()
        fn()
        times[i] = (time.perf_counter_ns() - t0) / 1e6
    mean = float(times.mean())
    return {
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
        "mean_ms": mean,
        "throughput_beats_per_s": 1000.0 / mean,
    }


def state_dict_bytes(model: torch.nn.Module) -> int:
    """Serialised size of a model's ``state_dict`` (the PyTorch FP32 'file size')."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes


def compact_metrics(m: dict[str, Any]) -> dict[str, Any]:
    """Headline metrics for the benchmark table."""
    return {
        "accuracy": m["accuracy"],
        "macro_f1": m["macro_f1"],
        "macro_f1_5class": m["macro_f1_5class"],
        "per_class_f1": {c: m["per_class"][c]["f1"] for c in AAMI_CLASSES},
        "per_class_precision": {c: m["per_class"][c]["precision"] for c in AAMI_CLASSES},
        "per_class_recall": {c: m["per_class"][c]["recall"] for c in AAMI_CLASSES},
    }


def benchmark(cfg: dict[str, Any]) -> dict[str, Any]:
    """Run the full benchmark and return the ``results/benchmark.json`` payload."""
    bcfg = cfg["benchmark"]
    threads = bcfg["num_threads"]
    torch.set_num_threads(threads)
    export = load_json(RESULTS_DIR / "export.json")
    best = load_json(RESULTS_DIR / "best_model.json")["variant"]
    ds = load_dataset()

    rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        model, ckpt = load_checkpoint(str(CHECKPOINT_DIR / f"inter_{variant}.pt"))
        test = ds.subset(inter_patient_splits(ds.record, ckpt["config"]["data"]["val_records"])["test"])
        beat, rr = test.x[:1], test.rr[:1]
        params = count_params(model)
        for runtime in RUNTIMES:
            if runtime == "pytorch_fp32":
                xb, rb = torch.from_numpy(beat), torch.from_numpy(rr)

                def fn(model: torch.nn.Module = model, xb: torch.Tensor = xb, rb: torch.Tensor = rb) -> None:
                    with torch.inference_mode():
                        model(xb, rb)

                size = state_dict_bytes(model)
                metrics = export[variant]["pytorch_fp32"]["test"]
                extra: dict[str, Any] = {"timed": "model(beat, rr) under torch.inference_mode()"}
            else:
                info = export[variant][runtime]
                clf = LiteRTClassifier(EXPORT_DIR / info["file"], num_threads=threads)
                clf.set_inputs(beat, rr)
                fn = clf.invoke
                size = info["size_bytes"]
                metrics = info["test"]
                extra = {
                    "timed": "interpreter.invoke() (inputs pre-set)",
                    "agreement_with_pytorch": info["agreement_with_pytorch"],
                    "io_dtype": info["io_dtype"],
                }
            lat = time_fn(fn, bcfg["warmup"], bcfg["runs"])
            row = {
                "variant": variant,
                "runtime": runtime,
                "runtime_label": RUNTIME_LABELS[runtime],
                "params": params,
                "size_bytes": size,
                "size_kb": size / 1024,
                **compact_metrics(metrics),
                "latency": lat,
                **extra,
            }
            rows.append(row)
            log.info(
                "%-5s %-20s %8.1fKB macroF1=%.4f p50=%.4fms p95=%.4fms",
                variant,
                runtime,
                row["size_kb"],
                row["macro_f1"],
                lat["p50_ms"],
                lat["p95_ms"],
            )

    def pick(v: str, rt: str) -> dict[str, Any]:
        return next(r for r in rows if r["variant"] == v and r["runtime"] == rt)

    fp32, int8, pt = pick(best, "litert_fp32"), pick(best, "litert_int8"), pick(best, "pytorch_fp32")
    quant_summary = {
        v: {
            "size_reduction_x": pick(v, "litert_fp32")["size_bytes"] / pick(v, "litert_int8")["size_bytes"],
            "macro_f1_delta": pick(v, "litert_int8")["macro_f1"] - pick(v, "litert_fp32")["macro_f1"],
            "accuracy_delta": pick(v, "litert_int8")["accuracy"] - pick(v, "litert_fp32")["accuracy"],
            "speedup_vs_litert_fp32_p50": pick(v, "litert_fp32")["latency"]["p50_ms"]
            / pick(v, "litert_int8")["latency"]["p50_ms"],
            "speedup_vs_pytorch_fp32_p50": pick(v, "pytorch_fp32")["latency"]["p50_ms"]
            / pick(v, "litert_int8")["latency"]["p50_ms"],
        }
        for v in VARIANTS
    }
    return {
        "hardware": hardware_info(threads),
        "protocol": "inter-patient (train DS1, test DS2 = 22 unseen patients, 49,668 beats)",
        "latency_settings": {"warmup": bcfg["warmup"], "runs": bcfg["runs"], "batch": 1, "threads": threads},
        "best_variant": best,
        "best_variant_selection": "highest validation macro-F1 (patient-wise DS1 hold-out); test set not used",
        "headline": {
            "variant": best,
            "runtime": "litert_int8",
            "accuracy": int8["accuracy"],
            "macro_f1": int8["macro_f1"],
            "size_kb": int8["size_kb"],
            "p50_ms": int8["latency"]["p50_ms"],
            "pytorch_macro_f1": pt["macro_f1"],
            "fp32_tflite_size_kb": fp32["size_kb"],
        },
        "quantization_summary": quant_summary,
        "rows": rows,
        "cpu_count": os.cpu_count(),
    }


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args(argv)
    save_json(benchmark(load_config(args.config)), RESULTS_DIR / "benchmark.json")
    log.info("wrote %s", RESULTS_DIR / "benchmark.json")


if __name__ == "__main__":
    main()
