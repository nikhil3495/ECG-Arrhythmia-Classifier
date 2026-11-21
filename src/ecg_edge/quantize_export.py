"""PyTorch -> LiteRT export with post-training int8 quantisation, plus numerical parity checks.

For every trained inter-patient variant this produces, in ``exports/``:

* ``<variant>_fp32.tflite``        - float model converted with ``litert_torch.convert``
* ``<variant>_int8.tflite``        - full-integer static int8 (int8 weights, activations *and* I/O),
  via litert-torch's PT2E quantiser calibrated on beats drawn from the TRAIN split only
* ``<variant>_int8_dynamic.tflite`` - dynamic-range int8 (int8 weights, float activations)

and checks every model against PyTorch on the full DS2 test set (``results/export.json``).
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ecg_edge.data import BeatDataset, inter_patient_splits, load_dataset
from ecg_edge.evaluate import compute_metrics, predict_torch
from ecg_edge.models import ECGNet, fold_rr_normalization, load_checkpoint
from ecg_edge.preprocess import RR_FEATURE_NAMES
from ecg_edge.runtime import LiteRTClassifier
from ecg_edge.utils import CHECKPOINT_DIR, EXPORT_DIR, RESULTS_DIR, get_logger, load_config, save_json, set_seed

log = get_logger(__name__)
VARIANTS = ("tiny", "small", "base")
RUNTIMES = ("litert_fp32", "litert_int8", "litert_int8_dynamic")


def sample_args(beat_len: int = 256, n_rr: int = len(RR_FEATURE_NAMES)) -> tuple[torch.Tensor, torch.Tensor]:
    """Batch-1 example inputs that fix the exported signature."""
    return torch.zeros(1, 1, beat_len), torch.ones(1, n_rr)


def export_fp32(model: ECGNet, path: Path) -> Path:
    """Convert an eval-mode model to a float32 LiteRT flatbuffer."""
    import litert_torch

    litert_torch.convert(model.eval(), sample_args()).export(str(path))
    return path


def export_int8(model: ECGNet, calib: BeatDataset, path: Path, dynamic: bool = False) -> Path:
    """PT2E post-training quantisation (per-channel symmetric weights) and LiteRT conversion.

    Args:
        model: Trained eval-mode model.
        calib: Representative beats (TRAIN split only); ignored for dynamic-range quantisation.
        path: Output ``.tflite`` path.
        dynamic: ``False`` -> static full-integer int8; ``True`` -> dynamic-range int8.

    """
    import litert_torch
    from litert_torch.quantize.pt2e_quantizer import PT2EQuantizer, get_symmetric_quantization_config
    from litert_torch.quantize.quant_config import QuantConfig
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    args = sample_args()
    quantizer = PT2EQuantizer().set_global(get_symmetric_quantization_config(is_per_channel=True, is_dynamic=dynamic))
    graph = torch.export.export(model.eval(), args).module()
    graph = prepare_pt2e(graph, quantizer)
    with torch.no_grad():
        for i in range(len(calib)):  # the exported graph is batch-1
            graph(torch.from_numpy(calib.x[i : i + 1]), torch.from_numpy(calib.rr[i : i + 1]))
    graph = convert_pt2e(graph, fold_quantize=False)
    litert_torch.convert(graph, args, quant_config=QuantConfig(pt2e_quantizer=quantizer)).export(str(path))
    return path


def calibration_set(train: BeatDataset, n: int, seed: int) -> BeatDataset:
    """Uniform random sample of ``n`` TRAIN beats (keeps the natural class mix the model will see)."""
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(train), size=min(n, len(train)), replace=False))
    return train.subset(idx)


def export_variant(variant: str, ds: BeatDataset, cfg: dict[str, Any]) -> dict[str, Any]:
    """Export one variant in all formats and evaluate each against PyTorch on DS2."""
    model, ckpt = load_checkpoint(str(CHECKPOINT_DIR / f"inter_{variant}.pt"))
    ccfg = ckpt["config"]
    splits = inter_patient_splits(ds.record, ccfg["data"]["val_records"])
    train, test = ds.subset(splits["train"]), ds.subset(splits["test"])
    calib = calibration_set(train, cfg["export"]["calibration_beats"], cfg["seed"])

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "litert_fp32": EXPORT_DIR / f"{variant}_fp32.tflite",
        "litert_int8": EXPORT_DIR / f"{variant}_int8.tflite",
        "litert_int8_dynamic": EXPORT_DIR / f"{variant}_int8_dynamic.tflite",
    }
    t0 = time.perf_counter()
    deploy = fold_rr_normalization(model)  # integer-only graph; PyTorch reference below stays unfolded
    export_fp32(deploy, paths["litert_fp32"])
    export_int8(deploy, calib, paths["litert_int8"], dynamic=False)
    export_int8(deploy, calib, paths["litert_int8_dynamic"], dynamic=True)
    log.info("%s: exported 3 models in %.1fs", variant, time.perf_counter() - t0)

    torch_logits = predict_torch(model, test.x, test.rr)
    torch_pred = torch_logits.argmax(1)
    out: dict[str, Any] = {
        "variant": variant,
        "calibration_beats": len(calib),
        "calibration_source": "random sample of the inter-patient TRAIN split (DS1 minus validation records)",
        "pytorch_fp32": {"test": compute_metrics(test.y, torch_pred)},
    }
    np.save(EXPORT_DIR / f"preds_{variant}_pytorch_fp32.npy", torch_pred)
    for rt, path in paths.items():
        clf = LiteRTClassifier(path)
        logits = clf.predict_logits(test.x, test.rr)
        pred = logits.argmax(1)
        np.save(EXPORT_DIR / f"preds_{variant}_{rt}.npy", pred)
        out[rt] = {
            "file": path.name,
            "size_bytes": path.stat().st_size,
            "io_dtype": str(clf.input_dtype),
            "agreement_with_pytorch": float((pred == torch_pred).mean()),
            "max_abs_logit_diff": float(np.abs(logits - torch_logits).max()),
            "test": compute_metrics(test.y, pred),
        }
        log.info(
            "%s %-20s size=%7.1fKB agree=%.5f macroF1=%.4f",
            variant,
            rt,
            path.stat().st_size / 1024,
            out[rt]["agreement_with_pytorch"],
            out[rt]["test"]["macro_f1"],
        )
    fp32_agree = out["litert_fp32"]["agreement_with_pytorch"]
    out["fp32_parity_ok"] = fp32_agree > 0.999
    if not out["fp32_parity_ok"]:
        log.error("%s: FP32 LiteRT agreement %.5f is below the 99.9%% requirement", variant, fp32_agree)
    return out


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    ds = load_dataset()
    results = {v: export_variant(v, ds, cfg) for v in args.variants}
    save_json(results, RESULTS_DIR / "export.json")
    if not all(r["fp32_parity_ok"] for r in results.values()):
        raise SystemExit("FP32 parity check failed - see results/export.json")


if __name__ == "__main__":
    main()
