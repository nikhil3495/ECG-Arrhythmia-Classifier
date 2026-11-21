"""Classification metrics and PyTorch-checkpoint evaluation on the held-out test split."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from ecg_edge import AAMI_CLASSES, SCORED_CLASSES
from ecg_edge.utils import CHECKPOINT_DIR, RESULTS_DIR, get_logger, save_json

log = get_logger(__name__)

SCORED_IDX = [AAMI_CLASSES.index(c) for c in SCORED_CLASSES]
ALL_IDX = list(range(len(AAMI_CLASSES)))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Accuracy, macro-F1 (N,S,V,F and all 5 classes), per-class P/R/F1 and the confusion matrix."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=ALL_IDX, zero_division=0)
    per_class = {
        c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
        for i, c in enumerate(AAMI_CLASSES)
    }
    return {
        "n": len(y_true),
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, labels=SCORED_IDX, average="macro", zero_division=0)),
        "macro_f1_5class": float(f1_score(y_true, y_pred, labels=ALL_IDX, average="macro", zero_division=0)),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=ALL_IDX).tolist(),
    }


def mean_f1(metrics: dict[str, Any], classes: Sequence[str]) -> float:
    """Unweighted mean F1 over ``classes`` (used as the early-stopping signal)."""
    return float(np.mean([metrics["per_class"][c]["f1"] for c in classes]))


@torch.no_grad()
def predict_torch(model: torch.nn.Module, x: np.ndarray, rr: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    """Return logits ``(n, 5)`` for numpy inputs using an eval-mode PyTorch model."""
    model.eval()
    out = []
    for i in range(0, len(x), batch_size):
        xb = torch.from_numpy(np.ascontiguousarray(x[i : i + batch_size]))
        rb = torch.from_numpy(np.ascontiguousarray(rr[i : i + batch_size]))
        out.append(model(xb, rb).numpy())
    return np.concatenate(out) if out else np.zeros((0, len(AAMI_CLASSES)), dtype=np.float32)


def main(argv: Sequence[str] | None = None) -> None:
    """CLI: evaluate PyTorch checkpoints on their protocol's test split."""
    from ecg_edge.data import inter_patient_splits, intra_patient_splits, load_dataset
    from ecg_edge.models import load_checkpoint

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", default=None, help="default: all checkpoints/*.pt")
    args = parser.parse_args(argv)
    paths = [Path(p) for p in args.checkpoints] if args.checkpoints else sorted(CHECKPOINT_DIR.glob("*.pt"))
    ds = load_dataset()
    for path in paths:
        model, ckpt = load_checkpoint(str(path))
        cfg = ckpt["config"]
        protocol = cfg["train"]["protocol"]
        if protocol == "inter":
            splits = inter_patient_splits(ds.record, cfg["data"]["val_records"])
        else:
            splits = intra_patient_splits(ds.y, cfg["seed"])
        test = ds.subset(splits["test"])
        metrics = compute_metrics(test.y, predict_torch(model, test.x, test.rr).argmax(1))
        out = {"checkpoint": path.name, "protocol": protocol, "variant": cfg["model"]["name"], "test": metrics}
        save_json(out, RESULTS_DIR / f"eval_{path.stem}.json")
        log.info(
            "%s: acc=%.4f macroF1(NSVF)=%.4f F1 %s",
            path.stem,
            metrics["accuracy"],
            metrics["macro_f1"],
            {c: round(v["f1"], 3) for c, v in metrics["per_class"].items()},
        )


if __name__ == "__main__":
    main()
