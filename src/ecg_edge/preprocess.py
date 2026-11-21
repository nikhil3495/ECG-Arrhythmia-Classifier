"""Signal preprocessing: AAMI label mapping, filtering, beat segmentation and RR features."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

from ecg_edge import AAMI_CLASSES

FS = 360
"""MIT-BIH sampling rate (Hz)."""

# AAMI EC57 mapping of MIT-BIH beat annotation symbols to the five superclasses.
AAMI_MAP: dict[str, str] = {
    # N: normal and bundle-branch-block beats, escape beats
    "N": "N",
    "L": "N",
    "R": "N",
    "e": "N",
    "j": "N",
    # S: supraventricular ectopic beats
    "A": "S",
    "a": "S",
    "J": "S",
    "S": "S",
    # V: ventricular ectopic beats
    "V": "V",
    "E": "V",
    # F: fusion of ventricular and normal
    "F": "F",
    # Q: paced, fusion of paced and normal, unclassifiable
    "/": "Q",
    "f": "Q",
    "Q": "Q",
}
CLASS_TO_INDEX: dict[str, int] = {c: i for i, c in enumerate(AAMI_CLASSES)}

RR_FEATURE_NAMES: tuple[str, ...] = (
    "pre_rr",
    "post_rr",
    "local_rr",
    "long_rr",
    "pre_over_local",
    "post_over_pre",
    "pre_over_long",
    "post_over_long",
)
"""Per-beat rhythm features (seconds / ratios). All use only past beats plus the next R-peak.

The ratios to ``long_rr`` (mean RR over the preceding 5 min) normalise for each patient's own
heart rate, which is what lets prematurity (the key S-beat cue) transfer to unseen patients.
"""


def symbol_to_class(symbol: str) -> int | None:
    """Map a MIT-BIH annotation symbol to an AAMI class index, or ``None`` for non-beat symbols."""
    cls = AAMI_MAP.get(symbol)
    return None if cls is None else CLASS_TO_INDEX[cls]


def bandpass(signal: np.ndarray, fs: int = FS, low: float = 0.5, high: float = 40.0, order: int = 3) -> np.ndarray:
    """Zero-phase Butterworth band-pass filter (removes baseline wander and HF noise)."""
    sos = butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, signal).astype(np.float32)


def znorm(beats: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Per-beat z-normalisation along the last axis."""
    mean = beats.mean(axis=-1, keepdims=True)
    std = beats.std(axis=-1, keepdims=True)
    return ((beats - mean) / (std + eps)).astype(np.float32)


def rr_features(r_peaks: np.ndarray, fs: int = FS, local_beats: int = 10, long_seconds: float = 300.0) -> np.ndarray:
    """Compute RR features for every beat; rows for the first and last beat contain NaN.

    Args:
        r_peaks: Sorted sample indices of *all* beat annotations in a record.
        fs: Sampling frequency.
        local_beats: Number of preceding RR intervals averaged for ``local_rr``.
        long_seconds: Window (s) of preceding RR intervals averaged for ``long_rr``.

    Returns:
        Array of shape ``(len(r_peaks), len(RR_FEATURE_NAMES))`` ordered as :data:`RR_FEATURE_NAMES`.

    """
    r = np.asarray(r_peaks, dtype=np.float64)
    n = len(r)
    feats = np.full((n, len(RR_FEATURE_NAMES)), np.nan, dtype=np.float64)
    if n < 3:
        return feats.astype(np.float32)
    rr = np.diff(r) / fs  # rr[i] = interval between beat i and i+1
    cumsum = np.concatenate([[0.0], np.cumsum(rr)])
    times = r / fs
    for i in range(1, n - 1):
        pre = rr[i - 1]
        post = rr[i]
        lo = max(0, i - local_beats)
        local = (cumsum[i] - cumsum[lo]) / (i - lo)
        j = int(np.searchsorted(times, times[i] - long_seconds, side="left"))
        j = min(j, i - 1)
        long_ = (cumsum[i] - cumsum[j]) / (i - j)
        feats[i] = (pre, post, local, long_, pre / local, post / pre, pre / long_, post / long_)
    return feats.astype(np.float32)


def segment_beats(
    signal: np.ndarray, r_peaks: np.ndarray, before: int = 90, after: int = 166
) -> tuple[np.ndarray, np.ndarray]:
    """Cut fixed windows ``[r - before, r + after)`` around each R-peak.

    Returns:
        ``(beats, valid)`` where ``beats`` has shape ``(n_valid, before + after)`` and ``valid`` is a
        boolean mask over ``r_peaks`` marking windows that fit inside the signal.

    """
    r_peaks = np.asarray(r_peaks, dtype=np.int64)
    valid = (r_peaks - before >= 0) & (r_peaks + after <= len(signal))
    idx = r_peaks[valid][:, None] + np.arange(-before, after)[None, :]
    return signal[idx].astype(np.float32), valid
