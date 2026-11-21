"""Validation-only hyper-parameter sweep on the tiny variant -> results/sweep_tiny.json.

Every run trains on the inter-patient TRAIN records and is scored on the patient-wise VAL records;
the DS2 test set is never touched. Runs whose result already exists in results/sweep/ are reused.
"""

from __future__ import annotations

import argparse
import copy
from typing import Any

import numpy as np

from ecg_edge import train as trainer
from ecg_edge.utils import CHECKPOINT_DIR, RESULTS_DIR, get_logger, load_config, load_json, save_json

log = get_logger("hparam_sweep")
SWEEP_DIR = RESULTS_DIR / "sweep"

RUNS: dict[str, dict[str, Any]] = {
    "w05": {"train.class_weight_power": 0.5},
    "w10": {"train.class_weight_power": 1.0},
    "w10_s1": {"train.class_weight_power": 1.0, "seed": 1},
    "w10_s2": {"train.class_weight_power": 1.0, "seed": 2},
    "w05_lr1e3_drop3_noise": {
        "train.lr": 0.001,
        "model.dropout": 0.3,
        "train.augment.noise_std": 0.1,
    },
    "w10_lr1e3": {"train.class_weight_power": 1.0, "train.lr": 0.001},
    "w10_reg": {"train.class_weight_power": 1.0, "model.dropout": 0.3, "train.augment.noise_std": 0.1},
    "w10_reg_s1": {
        "train.class_weight_power": 1.0,
        "model.dropout": 0.3,
        "train.augment.noise_std": 0.1,
        "seed": 1,
    },
}


def apply(cfg: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``cfg`` with dotted-key overrides applied."""
    cfg = copy.deepcopy(cfg)
    for key, value in overrides.items():
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node[p]
        node[leaf] = value
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/tiny.yaml")
    parser.add_argument("--epochs", type=int, default=25)
    args = parser.parse_args()
    base = load_config(args.config)
    base["train"]["epochs"] = args.epochs
    base["train"]["class_weight_power"] = 0.5  # sweep baseline (pre-sweep default)
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, overrides in RUNS.items():
        out = SWEEP_DIR / f"{name}.json"
        if not out.exists():
            cfg = apply(base, overrides)
            cfg["model"]["name"] = f"sweep_{name}"
            summary = trainer.train(cfg)
            (CHECKPOINT_DIR / f"inter_sweep_{name}.pt").unlink(missing_ok=True)
            (RESULTS_DIR / f"train_inter_sweep_{name}.json").unlink(missing_ok=True)
            save_json({"overrides": overrides, **summary}, out)
        res = load_json(out)
        pc = res["val"]["per_class"]
        rows.append(
            {
                "run": name,
                "overrides": res["overrides"],
                "best_epoch": res["best_epoch"],
                "val_select_f1_NSV": res["val_select_f1"],
                "val_macro_f1_NSVF": res["val"]["macro_f1"],
                "val_f1": {c: pc[c]["f1"] for c in ("N", "S", "V", "F")},
            }
        )
        log.info("%-24s val NSV-F1=%.4f S-F1=%.3f", name, rows[-1]["val_select_f1_NSV"], rows[-1]["val_f1"]["S"])

    groups: dict[str, list[float]] = {}
    for r in rows:
        groups.setdefault(r["run"].split("_s")[0], []).append(r["val_select_f1_NSV"])
    save_json(
        {
            "variant": "tiny",
            "metric": "mean val F1 over N,S,V (patient-wise DS1 hold-out)",
            "runs": rows,
            "seed_means": {k: {"mean": float(np.mean(v)), "n_seeds": len(v)} for k, v in groups.items()},
        },
        RESULTS_DIR / "sweep_tiny.json",
    )


if __name__ == "__main__":
    main()
