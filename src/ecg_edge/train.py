"""Train an :class:`~ecg_edge.models.ECGNet` variant with early stopping on patient-wise validation F1."""

from __future__ import annotations

import argparse
import copy
import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ecg_edge import AAMI_CLASSES
from ecg_edge.data import BeatDataset, inter_patient_splits, intra_patient_splits, load_dataset
from ecg_edge.evaluate import compute_metrics, mean_f1, predict_torch
from ecg_edge.models import build_model, count_params
from ecg_edge.utils import CHECKPOINT_DIR, RESULTS_DIR, get_logger, load_config, save_json, set_seed

log = get_logger(__name__)


def class_weights(y: np.ndarray, power: float, max_weight: float) -> torch.Tensor:
    """Loss weights ``(n_max / n_c) ** power`` clipped to ``max_weight`` (classes absent -> max)."""
    counts = np.bincount(y, minlength=len(AAMI_CLASSES)).astype(np.float64)
    w = np.where(counts > 0, (counts.max() / np.maximum(counts, 1)) ** power, max_weight)
    return torch.tensor(np.minimum(w, max_weight), dtype=torch.float32)


def augment(x: torch.Tensor, aug: dict[str, float], gen: torch.Generator) -> torch.Tensor:
    """Random amplitude scaling, Gaussian noise and small circular time shift (train only)."""
    b = x.shape[0]
    scale = 1 + (torch.rand(b, 1, 1, generator=gen) * 2 - 1) * aug["scale"]
    x = x * scale + torch.randn(x.shape, generator=gen) * aug["noise_std"]
    shift = int(aug["shift"])
    if shift > 0:
        k = int(torch.randint(-shift, shift + 1, (1,), generator=gen))
        x = torch.roll(x, k, dims=-1)
    return x


def select_splits(ds: BeatDataset, cfg: dict[str, Any]) -> dict[str, np.ndarray]:
    """Index splits for the configured protocol."""
    if cfg["train"]["protocol"] == "inter":
        return inter_patient_splits(ds.record, cfg["data"]["val_records"])
    return intra_patient_splits(ds.y, cfg["seed"])


def train(cfg: dict[str, Any]) -> dict[str, Any]:
    """Train one model; save the best checkpoint and the training history. Returns a summary."""
    tcfg = cfg["train"]
    set_seed(cfg["seed"])
    torch.set_num_threads(tcfg.get("num_threads", 2))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = load_dataset()
    splits = select_splits(ds, cfg)
    tr, va = ds.subset(splits["train"]), ds.subset(splits["val"])
    log.info("protocol=%s train=%d val=%d", tcfg["protocol"], len(tr), len(va))

    model = build_model(cfg["model"])
    model.set_rr_stats(torch.from_numpy(tr.rr.mean(0)), torch.from_numpy(tr.rr.std(0)))
    model.to(device)
    n_params = count_params(model)
    log.info("model=%s params=%d", cfg["model"]["name"], n_params)

    weights = class_weights(tr.y, tcfg["class_weight_power"], tcfg["max_class_weight"]).to(device)
    log.info("class weights %s", dict(zip(AAMI_CLASSES, weights.tolist(), strict=True)))
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tcfg["epochs"])

    gen = torch.Generator().manual_seed(cfg["seed"])
    loader = DataLoader(
        TensorDataset(torch.from_numpy(tr.x), torch.from_numpy(tr.rr), torch.from_numpy(tr.y)),
        batch_size=tcfg["batch_size"],
        shuffle=True,
        generator=gen,
        drop_last=True,
    )

    select_classes = tcfg.get("select_classes", ["N", "S", "V", "F"])
    best_f1, best_state, best_epoch, bad = -1.0, None, -1, 0
    history: list[dict[str, float]] = []
    t0 = time.perf_counter()
    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        running = 0.0
        for xb, rb, yb in loader:
            xb = augment(xb, tcfg["augment"], gen)
            xb, rb, yb = xb.to(device), rb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb, rb), yb)
            loss.backward()
            opt.step()
            running += loss.item() * len(yb)
        sched.step()
        model.cpu()
        val = compute_metrics(va.y, predict_torch(model, va.x, va.rr).argmax(1))
        model.to(device)
        rec = {
            "epoch": epoch,
            "train_loss": running / (len(loader) * tcfg["batch_size"]),
            "val_accuracy": val["accuracy"],
            "val_macro_f1": val["macro_f1"],
            "val_select_f1": mean_f1(val, select_classes),
        }
        history.append(rec)
        log.info(
            "epoch %02d loss=%.4f val_acc=%.4f val_macroF1=%.4f val_selectF1=%.4f (%.0fs)",
            epoch,
            rec["train_loss"],
            rec["val_accuracy"],
            rec["val_macro_f1"],
            rec["val_select_f1"],
            time.perf_counter() - t0,
        )
        if rec["val_select_f1"] > best_f1:
            best_f1, best_epoch, bad = rec["val_select_f1"], epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= tcfg["patience"]:
                log.info("early stopping at epoch %d (best %d)", epoch, best_epoch)
                break

    train_seconds = time.perf_counter() - t0
    assert best_state is not None
    model.load_state_dict(best_state)
    model.cpu().eval()
    val_best = compute_metrics(va.y, predict_torch(model, va.x, va.rr).argmax(1))

    tag = f"{tcfg['protocol']}_{cfg['model']['name']}"
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / f"{tag}.pt"
    torch.save({"config": cfg, "state_dict": model.state_dict(), "best_epoch": best_epoch}, ckpt_path)
    summary = {
        "tag": tag,
        "variant": cfg["model"]["name"],
        "protocol": tcfg["protocol"],
        "params": n_params,
        "best_epoch": best_epoch,
        "select_classes": select_classes,
        "val_select_f1": best_f1,
        "epochs_run": len(history),
        "train_seconds": train_seconds,
        "device": str(device),
        "torch_threads": torch.get_num_threads(),
        "class_weights": weights.cpu().tolist(),
        "val": val_best,
        "history": history,
    }
    save_json(summary, RESULTS_DIR / f"train_{tag}.json")
    log.info("saved %s (best val %s-F1 %.4f @ epoch %d)", ckpt_path, "".join(select_classes), best_f1, best_epoch)
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="variant config, e.g. configs/small.yaml")
    parser.add_argument("--protocol", choices=["inter", "intra"], default=None, help="override train.protocol")
    parser.add_argument("--epochs", type=int, default=None, help="override train.epochs (smoke tests)")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.protocol:
        cfg["train"]["protocol"] = args.protocol
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    train(cfg)


if __name__ == "__main__":
    main()
