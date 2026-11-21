"""Bundle demo assets: the best int8 model plus 10-s DS2 (unseen-patient) ECG segments.

Writes ``app/models/ecg_int8.tflite``, ``app/models/model_card.json``, ``app/samples.npz`` (Gradio app)
and ``docs/demo/`` (in-browser LiteRT.js demo: model + samples.json incl. Python reference predictions).
"""

from __future__ import annotations

import shutil

import numpy as np

from ecg_edge import AAMI_CLASSES
from ecg_edge.data import DS2, load_dataset, load_record
from ecg_edge.preprocess import FS, bandpass
from ecg_edge.runtime import LiteRTClassifier, softmax
from ecg_edge.utils import EXPORT_DIR, PROJECT_ROOT, RESULTS_DIR, get_logger, load_config, load_json, save_json

log = get_logger("make_demo_assets")
SEG_SECONDS = 10
APP_DIR = PROJECT_ROOT / "app"
WEB_DIR = PROJECT_ROOT / "docs" / "demo"


def pick_segments(ds, per_class: dict[str, int]) -> list[tuple[int, int, str]]:
    """Deterministically choose 10-s windows from distinct DS2 records rich in each target class."""
    win = SEG_SECONDS * FS
    cands = []
    for rec in DS2:
        m = ds.record == rec
        samp, y = ds.sample[m], ds.y[m]
        for start in range(win, samp.max() - win, win):
            inwin = (samp >= start + 128) & (samp < start + win - 200)
            if inwin.sum() < 5:
                continue
            counts = np.bincount(y[inwin], minlength=len(AAMI_CLASSES))
            cands.append((rec, start, counts))
    chosen: list[tuple[int, int, str]] = []
    used: set[int] = set()
    taken: dict[str, int] = {}
    for cls, k in per_class.items():
        ci = AAMI_CLASSES.index(cls)
        if cls == "N":
            ranked = sorted((c for c in cands if c[2][1:].sum() == 0), key=lambda c: (c[0], c[1]))
        else:
            ranked = sorted(cands, key=lambda c: (-c[2][ci], c[0], c[1]))
        for rec, start, counts in ranked:
            if taken.get(cls, 0) >= k:
                break
            if rec in used or (cls != "N" and counts[ci] == 0):
                continue
            present = ", ".join(f"{AAMI_CLASSES[j]}x{int(n)}" for j, n in enumerate(counts) if n)
            chosen.append((rec, start, f"{cls}-rich ({present})" if cls != "N" else f"normal rhythm ({present})"))
            used.add(rec)
            taken[cls] = taken.get(cls, 0) + 1
    return chosen


def main() -> None:
    cfg = load_config("configs/default.yaml")
    best = load_json(RESULTS_DIR / "best_model.json")["variant"]
    bench = load_json(RESULTS_DIR / "benchmark.json")
    src_model = EXPORT_DIR / f"{best}_int8.tflite"
    ds = load_dataset()

    segments = pick_segments(ds, {"S": 4, "V": 3, "F": 2, "N": 2})
    signals, rec_ids, starts, labels, offsets = [], [], [], [], [0]
    pos_all, x_all, rr_all, y_all = [], [], [], []
    win = SEG_SECONDS * FS
    for rec, start, lab in segments:
        sig, _, _ = load_record(rec, lead=cfg["data"]["lead"])
        filt = bandpass(sig, low=cfg["data"]["filter"]["low_hz"], high=cfg["data"]["filter"]["high_hz"])
        signals.append(filt[start : start + win])
        m = (ds.record == rec) & (ds.sample >= start) & (ds.sample < start + win)
        idx = np.flatnonzero(m)
        pos_all.append(ds.sample[idx] - start)
        x_all.append(ds.x[idx])
        rr_all.append(ds.rr[idx])
        y_all.append(ds.y[idx])
        offsets.append(offsets[-1] + len(idx))
        rec_ids.append(rec)
        starts.append(start / FS)
        labels.append(lab)
        log.info("segment record %d @ %.0fs: %s", rec, start / FS, lab)

    beat_x = np.concatenate(x_all).astype(np.float32)
    beat_rr = np.concatenate(rr_all).astype(np.float32)
    beat_y = np.concatenate(y_all).astype(np.int64)

    (APP_DIR / "models").mkdir(parents=True, exist_ok=True)
    shutil.copy(src_model, APP_DIR / "models" / "ecg_int8.tflite")
    np.savez_compressed(
        APP_DIR / "samples.npz",
        fs=np.float32(FS),
        signals=np.stack(signals).astype(np.float32),
        seg_record=np.array(rec_ids),
        seg_start_s=np.array(starts, dtype=np.float32),
        seg_label=np.array(labels),
        beat_offsets=np.array(offsets),
        beat_pos=np.concatenate(pos_all),
        beat_x=beat_x,
        beat_rr=beat_rr,
        beat_y=beat_y,
    )
    save_json(
        {"variant": best, "source": src_model.name, "headline": bench["headline"], "classes": list(AAMI_CLASSES)},
        APP_DIR / "models" / "model_card.json",
    )

    # Web demo: same model + JSON samples + Python LiteRT reference outputs for an in-browser parity check.
    clf = LiteRTClassifier(src_model)
    web_x, web_rr = np.round(beat_x, 4), np.round(beat_rr, 5)  # exactly what the browser receives
    ref_probs = softmax(clf.predict_logits(web_x, web_rr))
    pos_cat = np.concatenate(pos_all)
    WEB_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(src_model, WEB_DIR / "ecg_int8.tflite")
    beat_d, rr_d, out_d = clf._beat, clf._rr, clf._out
    web = {
        "model": "ecg_int8.tflite",
        "variant": best,
        "classes": list(AAMI_CLASSES),
        "fs": FS,
        "quant": {
            "beat": {"shape": beat_d["shape"].tolist(), "scale": float(beat_d["quantization"][0]),
                     "zero_point": int(beat_d["quantization"][1])},
            "rr": {"shape": rr_d["shape"].tolist(), "scale": float(rr_d["quantization"][0]),
                   "zero_point": int(rr_d["quantization"][1])},
            "out": {"scale": float(out_d["quantization"][0]), "zero_point": int(out_d["quantization"][1])},
        },
        "headline": bench["headline"],
        "segments": [],
    }  # fmt: skip
    for i in range(len(segments)):
        lo, hi = offsets[i], offsets[i + 1]
        web["segments"].append(
            {
                "record": int(rec_ids[i]),
                "start_s": float(starts[i]),
                "label": labels[i],
                "signal": np.round(signals[i], 3).tolist(),
                "beats": [
                    {
                        "pos": int(pos_cat[k]),
                        "y": int(beat_y[k]),
                        "x": web_x[k, 0].tolist(),
                        "rr": web_rr[k].tolist(),
                        "ref_probs": np.round(ref_probs[k], 4).tolist(),
                        "ref_pred": int(ref_probs[k].argmax()),
                    }
                    for k in range(lo, hi)
                ],
            }
        )
    save_json(web, WEB_DIR / "samples.json")
    log.info(
        "wrote app + web demo assets: %d segments, %d beats, model %.1f KB",
        len(segments),
        len(beat_y),
        src_model.stat().st_size / 1024,
    )


if __name__ == "__main__":
    main()
