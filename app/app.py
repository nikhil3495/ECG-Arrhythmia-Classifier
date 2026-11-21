"""Gradio demo: classify heartbeats from unseen-patient (DS2) ECG segments with the int8 LiteRT model.

Only the full-integer int8 ``.tflite`` is loaded - no PyTorch at inference time.
Run: ``python app/app.py`` (or ``make app``).
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR.parent / "src"))  # allow running from a source checkout

from ecg_edge.runtime import LiteRTClassifier, softmax  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("ecg-demo")

CLASSES = ("N", "S", "V", "F", "Q")
CLASS_NAMES = {
    "N": "Normal / bundle-branch block",
    "S": "Supraventricular ectopic",
    "V": "Ventricular ectopic",
    "F": "Fusion (V + N)",
    "Q": "Unknown / paced",
}
CLASS_COLORS = {"N": "#2a78d6", "S": "#eb6834", "V": "#1baf7a", "F": "#eda100", "Q": "#e87ba4"}
MODEL_PATH = APP_DIR / "models" / "ecg_int8.tflite"
SAMPLES_PATH = APP_DIR / "samples.npz"
META_PATH = APP_DIR / "models" / "model_card.json"

DISCLAIMER = (
    "**Medical disclaimer:** research and education only. This is **not a medical device**, has not been "
    "clinically validated, and must not be used for diagnosis or treatment decisions."
)

clf = LiteRTClassifier(MODEL_PATH, num_threads=1)
samples = np.load(SAMPLES_PATH)
meta = json.loads(META_PATH.read_text(encoding="utf-8")) if META_PATH.exists() else {}
SEGMENTS = [
    f"#{i:02d} record {int(r)} @ {s:.0f}s - {lab}"
    for i, (r, s, lab) in enumerate(
        zip(samples["seg_record"], samples["seg_start_s"], samples["seg_label"], strict=True)
    )
]


def _segment(i: int) -> dict[str, np.ndarray]:
    lo, hi = samples["beat_offsets"][i], samples["beat_offsets"][i + 1]
    return {
        "signal": samples["signals"][i],
        "pos": samples["beat_pos"][lo:hi],
        "x": samples["beat_x"][lo:hi],
        "rr": samples["beat_rr"][lo:hi],
        "y": samples["beat_y"][lo:hi],
    }


def _plot(seg: dict[str, np.ndarray], pred: np.ndarray, fs: float) -> plt.Figure:
    t = np.arange(len(seg["signal"])) / fs
    fig, ax = plt.subplots(figsize=(11, 3.4))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.plot(t, seg["signal"], color="#52514e", lw=1)
    top = float(seg["signal"].max())
    for p, yp, yt in zip(seg["pos"], pred, seg["y"], strict=True):
        c = CLASSES[yp]
        ax.axvline(p / fs, color=CLASS_COLORS[c], alpha=0.25, lw=6)
        mark = c if yp == yt else f"{c}✗"
        ax.text(
            p / fs, top * 1.08, mark, ha="center", va="bottom", fontsize=10, fontweight="bold", color=CLASS_COLORS[c]
        )
    ax.set_ylim(top=top * 1.3 if top > 0 else 1)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("MLII (mV, band-passed)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("Predicted class above each beat (✗ = differs from cardiologist label)", loc="left", fontsize=10)
    fig.tight_layout()
    return fig


def classify(choice: str) -> tuple[plt.Figure, list[list[str]], dict[str, float], str]:
    """Run the int8 model on every beat of the chosen segment."""
    i = SEGMENTS.index(choice)
    seg = _segment(i)
    n = len(seg["y"])
    probs = np.empty((n, len(CLASSES)), dtype=np.float32)
    times = np.empty(n)
    for k in range(n):
        clf.set_inputs(seg["x"][k : k + 1], seg["rr"][k : k + 1])
        t0 = time.perf_counter()
        clf.invoke()
        times[k] = (time.perf_counter() - t0) * 1e3
        probs[k] = softmax(clf.output_logits())
    pred = probs.argmax(1)
    rows = [
        [str(k + 1), f"{seg['pos'][k] / float(samples['fs']):.2f}", CLASSES[seg["y"][k]], CLASSES[pred[k]]]
        + [f"{p:.3f}" for p in probs[k]]
        for k in range(n)
    ]
    # Show the probabilities of the most "interesting" beat: the first non-N prediction, else beat 1.
    focus = int(np.flatnonzero(pred != 0)[0]) if (pred != 0).any() else 0
    label = {f"{c} - {CLASS_NAMES[c]}": float(probs[focus, j]) for j, c in enumerate(CLASSES)}
    acc = float((pred == seg["y"]).mean())
    stats = (
        f"**{n} beats** classified - agreement with reference labels: **{acc:.0%}**  \n"
        f"int8 LiteRT `invoke()` latency on this server: median **{np.median(times):.3f} ms**/beat "
        f"(max {times.max():.3f} ms), 1 thread.  \n"
        f"Probabilities shown for beat **#{focus + 1}**."
    )
    return _plot(seg, pred, float(samples["fs"])), rows, label, stats


def build_ui() -> gr.Blocks:
    """Assemble the Gradio interface."""
    headline = meta.get("headline", {})
    info = (
        f"Model: **{meta.get('variant', '?')}** 1D-CNN + RR features, full-integer int8 LiteRT "
        f"({MODEL_PATH.stat().st_size / 1024:.1f} KB). "
        + (
            f"Inter-patient DS2 test: accuracy {headline['accuracy']:.3f}, macro-F1 (N,S,V,F) "
            f"{headline['macro_f1']:.3f}."
            if headline
            else ""
        )
    )
    with gr.Blocks(title="On-device ECG arrhythmia classifier") as demo:
        gr.Markdown(
            "# On-device ECG arrhythmia classifier (int8 LiteRT)\n"
            "Pick a 10-second MLII segment from a **DS2 test patient the model never saw** and classify every "
            "beat into the AAMI classes N / S / V / F / Q.\n\n" + info + "\n\n" + DISCLAIMER
        )
        with gr.Row():
            choice = gr.Dropdown(SEGMENTS, value=SEGMENTS[0], label="ECG segment (MIT-BIH DS2)", scale=4)
            btn = gr.Button("Classify", variant="primary", scale=1)
        plot = gr.Plot(label="ECG")
        stats = gr.Markdown()
        with gr.Row():
            table = gr.Dataframe(
                headers=["beat", "t (s)", "true", "pred", *[f"p({c})" for c in CLASSES]],
                label="Per-beat predictions",
                wrap=True,
            )
            probs = gr.Label(num_top_classes=5, label="Class probabilities")
        btn.click(classify, inputs=choice, outputs=[plot, table, probs, stats])
        choice.change(classify, inputs=choice, outputs=[plot, table, probs, stats])
        demo.load(classify, inputs=choice, outputs=[plot, table, probs, stats])
    return demo


if __name__ == "__main__":
    log.info("loaded %s (%d segments)", MODEL_PATH, len(SEGMENTS))
    build_ui().launch(server_name="0.0.0.0", server_port=7860)
