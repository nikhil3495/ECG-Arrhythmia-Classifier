"""MIT-BIH download, beat dataset construction, patient-wise splits and the data card.

Inter-patient protocol (de Chazal et al., 2004): DS1 records train the model, DS2 records test it,
so no patient contributes beats to both. Paced records (102, 104, 107, 217) are excluded per
AAMI EC57. A validation set is carved out of DS1 *by record* to keep model selection patient-wise.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ecg_edge import AAMI_CLASSES
from ecg_edge.preprocess import (
    FS,
    RR_FEATURE_NAMES,
    bandpass,
    rr_features,
    segment_beats,
    symbol_to_class,
    znorm,
)
from ecg_edge.utils import (
    PROCESSED_DIR,
    PROJECT_ROOT,
    RAW_DIR,
    RESULTS_DIR,
    get_logger,
    load_config,
    save_json,
)

log = get_logger(__name__)

# fmt: off
DS1: tuple[int, ...] = (
    101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122,
    124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230,
)
DS2: tuple[int, ...] = (
    100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210,
    212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234,
)
# fmt: on
PACED: tuple[int, ...] = (102, 104, 107, 217)
ALL_RECORDS: tuple[int, ...] = tuple(sorted(DS1 + DS2 + PACED))
DATASET_FILE = "beats.npz"


@dataclass
class BeatDataset:
    """In-memory beat dataset.

    Attributes:
        x: Beats, shape ``(n, 1, window)``, float32, z-normalised.
        rr: RR features, shape ``(n, n_rr)``, float32 (raw seconds / ratios).
        y: AAMI class indices, shape ``(n,)``, int64.
        record: Source MIT-BIH record number per beat, shape ``(n,)``.
        sample: R-peak sample index within its record, shape ``(n,)``.

    """

    x: np.ndarray
    rr: np.ndarray
    y: np.ndarray
    record: np.ndarray
    sample: np.ndarray

    def __len__(self) -> int:
        return len(self.y)

    def subset(self, idx: np.ndarray) -> BeatDataset:
        """Return the beats selected by ``idx``."""
        return BeatDataset(self.x[idx], self.rr[idx], self.y[idx], self.record[idx], self.sample[idx])


def download_mitdb(raw_dir: Path = RAW_DIR) -> Path:
    """Download the MIT-BIH Arrhythmia Database from PhysioNet if it is not cached."""
    import wfdb

    raw_dir.mkdir(parents=True, exist_ok=True)
    missing = [r for r in ALL_RECORDS if not all((raw_dir / f"{r}.{ext}").exists() for ext in ("dat", "hea", "atr"))]
    if missing:
        log.info("Downloading %d MIT-BIH records to %s", len(missing), raw_dir)
        wfdb.dl_database("mitdb", dl_dir=str(raw_dir), records=[str(r) for r in missing])
    else:
        log.info("MIT-BIH already present in %s", raw_dir)
    return raw_dir


def load_record(record: int, raw_dir: Path = RAW_DIR, lead: str = "MLII") -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Read one record's lead signal and its beat annotations."""
    import wfdb

    path = str(raw_dir / str(record))
    rec = wfdb.rdrecord(path)
    if lead not in rec.sig_name:
        raise ValueError(f"Record {record} has no {lead} lead (has {rec.sig_name})")
    signal = rec.p_signal[:, rec.sig_name.index(lead)].astype(np.float32)
    ann = wfdb.rdann(path, "atr")
    return signal, np.asarray(ann.sample, dtype=np.int64), list(ann.symbol)


def process_record(
    signal: np.ndarray,
    samples: np.ndarray,
    symbols: Sequence[str],
    before: int,
    after: int,
    filter_cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Turn one record into ``(beats, rr, labels, r_samples)`` for all AAMI-labelled beats."""
    labels_all = [symbol_to_class(s) for s in symbols]
    is_beat = np.array([lab is not None for lab in labels_all], dtype=bool)
    r_peaks = samples[is_beat]
    labels = np.array([lab for lab in labels_all if lab is not None], dtype=np.int64)

    filtered = bandpass(signal, fs=FS, low=filter_cfg["low_hz"], high=filter_cfg["high_hz"], order=filter_cfg["order"])
    rr = rr_features(r_peaks, fs=FS)
    beats, valid = segment_beats(filtered, r_peaks, before=before, after=after)
    keep = valid & ~np.isnan(rr).any(axis=1)
    beats = beats[keep[valid]]
    return znorm(beats), rr[keep], labels[keep], r_peaks[keep]


def build_dataset(cfg: dict[str, Any], raw_dir: Path = RAW_DIR, records: Iterable[int] | None = None) -> BeatDataset:
    """Preprocess the non-paced MIT-BIH records into a :class:`BeatDataset`."""
    dcfg = cfg["data"]
    records = list(records) if records is not None else sorted(DS1 + DS2)
    xs, rrs, ys, recs, samps = [], [], [], [], []
    for rec in records:
        signal, samples, symbols = load_record(rec, raw_dir, dcfg["lead"])
        b, rr, y, s = process_record(signal, samples, symbols, dcfg["before"], dcfg["after"], dcfg["filter"])
        xs.append(b)
        rrs.append(rr)
        ys.append(y)
        recs.append(np.full(len(y), rec, dtype=np.int64))
        samps.append(s)
        log.debug("record %s: %d beats", rec, len(y))
    return BeatDataset(
        x=np.concatenate(xs)[:, None, :],
        rr=np.concatenate(rrs),
        y=np.concatenate(ys),
        record=np.concatenate(recs),
        sample=np.concatenate(samps),
    )


def save_dataset(ds: BeatDataset, path: Path) -> None:
    """Cache a dataset as compressed npz."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, x=ds.x, rr=ds.rr, y=ds.y, record=ds.record, sample=ds.sample)


def load_dataset(path: Path = PROCESSED_DIR / DATASET_FILE) -> BeatDataset:
    """Load the cached dataset produced by ``make data``."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `make data` first")
    with np.load(path) as z:
        return BeatDataset(z["x"], z["rr"], z["y"], z["record"], z["sample"])


def inter_patient_splits(record: np.ndarray, val_records: Sequence[int]) -> dict[str, np.ndarray]:
    """Index arrays for the patient-wise protocol: train/val from DS1 (by record), test = DS2."""
    val_records = set(val_records)
    if not val_records <= set(DS1):
        raise ValueError("Validation records must come from DS1")
    is_val = np.isin(record, list(val_records))
    is_ds1 = np.isin(record, DS1)
    return {
        "train": np.flatnonzero(is_ds1 & ~is_val),
        "val": np.flatnonzero(is_val),
        "test": np.flatnonzero(np.isin(record, DS2)),
    }


def intra_patient_splits(
    y: np.ndarray, seed: int, test_frac: float = 0.2, val_frac: float = 0.1
) -> dict[str, np.ndarray]:
    """Random beat-level stratified split. Beats of one patient land in train and test (leaky)."""
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(y))
    trainval, test = train_test_split(idx, test_size=test_frac, random_state=seed, stratify=y)
    train, val = train_test_split(trainval, test_size=val_frac, random_state=seed, stratify=y[trainval])
    return {"train": np.sort(train), "val": np.sort(val), "test": np.sort(test)}


def class_counts(y: np.ndarray) -> dict[str, int]:
    """Per-AAMI-class beat counts."""
    counts = np.bincount(y, minlength=len(AAMI_CLASSES))
    return {c: int(n) for c, n in zip(AAMI_CLASSES, counts, strict=True)}


def data_card(ds: BeatDataset, cfg: dict[str, Any]) -> dict[str, Any]:
    """Summarise per-split class counts and records for both protocols."""
    inter = inter_patient_splits(ds.record, cfg["data"]["val_records"])
    intra = intra_patient_splits(ds.y, cfg["seed"])
    card: dict[str, Any] = {
        "source": "MIT-BIH Arrhythmia Database (PhysioNet mitdb), lead MLII, 360 Hz",
        "excluded_paced_records": list(PACED),
        "window_samples": int(ds.x.shape[-1]),
        "window": {"before": cfg["data"]["before"], "after": cfg["data"]["after"]},
        "rr_features": list(RR_FEATURE_NAMES),
        "total_beats": len(ds),
        "inter_patient": {},
        "intra_patient": {},
    }
    for name, idx in inter.items():
        recs = sorted({int(r) for r in ds.record[idx]})
        card["inter_patient"][name] = {"records": recs, "n_beats": len(idx), "class_counts": class_counts(ds.y[idx])}
    for name, idx in intra.items():
        card["intra_patient"][name] = {"n_beats": len(idx), "class_counts": class_counts(ds.y[idx])}
    return card


def write_data_card_md(card: dict[str, Any], path: Path) -> None:
    """Render the data card JSON as Markdown."""
    lines = [
        "# Data card - MIT-BIH beats",
        "",
        "_Auto-generated by `python -m ecg_edge.data`; do not edit by hand._",
        "",
        f"- **Source:** {card['source']}",
        f"- **Excluded (paced, AAMI EC57):** {', '.join(map(str, card['excluded_paced_records']))}",
        f"- **Beat window:** {card['window']['before']} samples before + {card['window']['after']} after "
        f"the R-peak = {card['window_samples']} samples ({card['window_samples'] / FS:.3f} s)",
        f"- **RR features:** {', '.join(card['rr_features'])}",
        f"- **Total beats:** {card['total_beats']:,}",
        "",
    ]
    for proto, title in (
        ("inter_patient", "Inter-patient (primary)"),
        ("intra_patient", "Intra-patient (secondary, leaky)"),
    ):
        lines += [
            f"## {title}",
            "",
            "| Split | Beats | " + " | ".join(AAMI_CLASSES) + " |",
            "|---|---:|" + "---:|" * len(AAMI_CLASSES),
        ]
        for split, info in card[proto].items():
            counts = " | ".join(f"{info['class_counts'][c]:,}" for c in AAMI_CLASSES)
            lines.append(f"| {split} | {info['n_beats']:,} | {counts} |")
        lines.append("")
        if proto == "inter_patient":
            for split, info in card[proto].items():
                lines.append(f"- **{split} records:** {', '.join(map(str, info['records']))}")
            lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    """CLI: download MIT-BIH, preprocess all non-paced records, cache them and write the data card."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--force", action="store_true", help="rebuild even if the cache exists")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    out = PROCESSED_DIR / DATASET_FILE
    if out.exists() and not args.force:
        log.info("Using cached %s", out)
        ds = load_dataset(out)
    else:
        download_mitdb()
        ds = build_dataset(cfg)
        save_dataset(ds, out)
        log.info("Saved %d beats to %s", len(ds), out)

    card = data_card(ds, cfg)
    save_json(card, RESULTS_DIR / "data_card.json")
    write_data_card_md(card, PROJECT_ROOT / "docs" / "DATA_CARD.md")
    for split, info in card["inter_patient"].items():
        log.info("inter-patient %-5s %6d beats %s", split, info["n_beats"], info["class_counts"])


if __name__ == "__main__":
    main()
