"""Figures for the README/report, rendered from results/*.json only."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ecg_edge import AAMI_CLASSES, SCORED_CLASSES
from ecg_edge.utils import FIGURES_DIR, RESULTS_DIR, get_logger, load_json

log = get_logger(__name__)

# Validated reference categorical palette (slots 1, 2, 3, 7) - fixed runtime -> colour mapping.
COLORS = {
    "pytorch_fp32": "#2a78d6",
    "litert_fp32": "#eb6834",
    "litert_int8": "#1baf7a",
    "litert_int8_dynamic": "#6250d6",
}
MARKERS = {"pytorch_fp32": "o", "litert_fp32": "s", "litert_int8": "D", "litert_int8_dynamic": "^"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SEQ_CMAP = "Blues"
VARIANTS = ("tiny", "small", "base")


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK2,
            "axes.titlecolor": INK,
            "axes.titleweight": "semibold",
            "axes.titlesize": 12,
            "axes.titlelocation": "left",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "font.family": "DejaVu Sans",
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
        }
    )


def _save(fig: plt.Figure, name: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path)
    plt.close(fig)
    log.info("wrote %s", path)
    return path


def _rows(bench: dict[str, Any], runtime: str) -> list[dict[str, Any]]:
    return [r for v in VARIANTS for r in bench["rows"] if r["variant"] == v and r["runtime"] == runtime]


def tradeoff(bench: dict[str, Any], x_key: str, xlabel: str, name: str, title: str) -> Path:
    """Macro-F1 vs ``x_key`` (size or latency), one line per runtime across tiny->small->base."""
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for rt, color in COLORS.items():
        rows = _rows(bench, rt)
        xs = [r["size_kb"] if x_key == "size_kb" else r["latency"]["p50_ms"] for r in rows]
        ys = [r["macro_f1"] for r in rows]
        ax.plot(xs, ys, color=color, lw=2, marker=MARKERS[rt], ms=8, mec=SURFACE, mew=2, label=rows[0]["runtime_label"])
        if rt == "litert_int8":
            for r, x, y in zip(rows, xs, ys, strict=True):
                ax.annotate(r["variant"], (x, y), xytext=(6, -12), textcoords="offset points", color=INK2, fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Macro-F1 (N, S, V, F) - inter-patient DS2")
    ax.set_title(title)
    ax.legend(loc="best")
    return _save(fig, name)


def confusion(bench: dict[str, Any], export: dict[str, Any], name: str) -> Path:
    """Row-normalised confusion matrix (with counts) for the best variant's int8 model."""
    best = bench["best_variant"]
    cm = np.array(export[best]["litert_int8"]["test"]["confusion_matrix"], dtype=float)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    ax.imshow(norm, cmap=SEQ_CMAP, vmin=0, vmax=1)
    ax.grid(False)
    for i in range(len(AAMI_CLASSES)):
        for j in range(len(AAMI_CLASSES)):
            color = "white" if norm[i, j] > 0.55 else INK
            ax.text(j, i, f"{norm[i, j]:.2f}\n({int(cm[i, j]):,})", ha="center", va="center", fontsize=8, color=color)
    ax.set_xticks(range(len(AAMI_CLASSES)), AAMI_CLASSES)
    ax.set_yticks(range(len(AAMI_CLASSES)), AAMI_CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix - {best}, LiteRT int8, DS2", fontsize=11)
    return _save(fig, name)


def per_class(bench: dict[str, Any], name: str) -> Path:
    """Per-class F1 bars for the best variant: PyTorch FP32 vs LiteRT FP32 vs LiteRT int8."""
    best = bench["best_variant"]
    rts = ("pytorch_fp32", "litert_fp32", "litert_int8")
    rows = {r["runtime"]: r for r in bench["rows"] if r["variant"] == best}
    x = np.arange(len(SCORED_CLASSES))
    w = 0.26
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for k, rt in enumerate(rts):
        vals = [rows[rt]["per_class_f1"][c] for c in SCORED_CLASSES]
        bars = ax.bar(x + (k - 1) * w, vals, w - 0.02, color=COLORS[rt], label=rows[rt]["runtime_label"])
        if rt == "litert_int8":
            ax.bar_label(bars, fmt="%.2f", fontsize=8, color=INK2, padding=2)
    ax.set_xticks(x, SCORED_CLASSES)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("F1 on DS2 (inter-patient)")
    ax.set_title(f"Per-class F1 - {best} model")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
    return _save(fig, name)


def fp32_vs_int8(bench: dict[str, Any], name: str) -> Path:
    """Three small multiples (size, macro-F1, latency) comparing LiteRT FP32 and int8 per variant."""
    metrics = (
        ("size_kb", "Model file size (KB)"),
        ("macro_f1", "Macro-F1 (N,S,V,F)"),
        ("p50_ms", "p50 latency (ms, host CPU)"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    x = np.arange(len(VARIANTS))
    w = 0.38
    for ax, (key, title) in zip(axes, metrics, strict=True):
        for k, rt in enumerate(("litert_fp32", "litert_int8")):
            rows = _rows(bench, rt)
            vals = [r["latency"]["p50_ms"] if key == "p50_ms" else r[key] for r in rows]
            bars = ax.bar(x + (k - 0.5) * w, vals, w - 0.03, color=COLORS[rt], label=rows[0]["runtime_label"])
            fmt = "%.3f" if key != "size_kb" else "%.0f"
            ax.bar_label(bars, fmt=fmt, fontsize=7, color=INK2, padding=2)
        ax.set_xticks(x, VARIANTS)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="x", visible=False)
        ax.margins(y=0.15)
    axes[0].legend(loc="upper left")
    fig.suptitle("LiteRT FP32 vs full-integer int8", x=0.01, ha="left", fontweight="semibold", color=INK)
    fig.tight_layout()
    return _save(fig, name)


def inter_vs_intra(name: str) -> Path | None:
    """Per-class F1 of the same architecture under the honest vs the leaky split."""
    best = load_json(RESULTS_DIR / "best_model.json")["variant"]
    inter_p, intra_p = RESULTS_DIR / f"eval_inter_{best}.json", RESULTS_DIR / f"eval_intra_{best}.json"
    if not (inter_p.exists() and intra_p.exists()):
        return None
    inter, intra = load_json(inter_p)["test"], load_json(intra_p)["test"]
    labels = [*SCORED_CLASSES, "macro"]
    vals = {
        "Inter-patient (DS1 -> DS2)": [inter["per_class"][c]["f1"] for c in SCORED_CLASSES] + [inter["macro_f1"]],
        "Intra-patient random split (leaky)": [intra["per_class"][c]["f1"] for c in SCORED_CLASSES]
        + [intra["macro_f1"]],
    }
    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for k, (lab, v) in enumerate(vals.items()):
        bars = ax.bar(x + (k - 0.5) * w, v, w - 0.03, color=("#2a78d6", "#eb6834")[k], label=lab)
        ax.bar_label(bars, fmt="%.2f", fontsize=8, color=INK2, padding=2)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("F1 (PyTorch FP32)")
    ax.set_title(f"Same {best} model, two evaluation protocols")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    return _save(fig, name)


def training_curves(name: str) -> Path:
    """Validation macro-F1 per epoch for the inter-patient runs."""
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for v, color in zip(VARIANTS, ("#2a78d6", "#eb6834", "#1baf7a"), strict=True):
        hist = load_json(RESULTS_DIR / f"train_inter_{v}.json")["history"]
        ax.plot([h["epoch"] for h in hist], [h["val_macro_f1"] for h in hist], color=color, lw=2, label=v)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation macro-F1")
    ax.set_title("Patient-wise validation macro-F1 (early-stopping signal)")
    ax.legend()
    return _save(fig, name)


def main(argv: Sequence[str] | None = None) -> None:
    """Render all figures."""
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    _style()
    bench = load_json(RESULTS_DIR / "benchmark.json")
    export = load_json(RESULTS_DIR / "export.json")
    tradeoff(bench, "size_kb", "Model file size (KB, log scale)", "f1_vs_size.png", "Macro-F1 vs model size")
    tradeoff(
        bench,
        "p50_ms",
        "p50 single-beat latency (ms, log scale) - desktop x86 CPU, 1 thread",
        "f1_vs_latency.png",
        "Macro-F1 vs latency",
    )
    confusion(bench, export, "confusion_int8_best.png")
    per_class(bench, "per_class_f1.png")
    fp32_vs_int8(bench, "fp32_vs_int8.png")
    inter_vs_intra("inter_vs_intra.png")
    training_curves("training_curves.png")


if __name__ == "__main__":
    main()
