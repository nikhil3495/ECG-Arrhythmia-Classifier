import numpy as np
import pytest

from ecg_edge.data import DS1, DS2, PACED, inter_patient_splits, intra_patient_splits, process_record
from ecg_edge.preprocess import FS
from ecg_edge.utils import load_config

VAL = load_config("configs/default.yaml")["data"]["val_records"]


def test_record_sets_are_disjoint_and_complete() -> None:
    assert len(DS1) == len(DS2) == 22
    assert not set(DS1) & set(DS2)
    assert not (set(DS1) | set(DS2)) & set(PACED)
    assert len(set(DS1) | set(DS2) | set(PACED)) == 48


def test_no_patient_overlap_between_splits() -> None:
    record = np.repeat(np.array(sorted(DS1 + DS2)), 50)
    s = inter_patient_splits(record, VAL)
    recs = {k: set(record[v].tolist()) for k, v in s.items()}
    assert not recs["train"] & recs["test"]
    assert not recs["train"] & recs["val"]
    assert not recs["val"] & recs["test"]
    assert recs["test"] == set(DS2)
    assert recs["train"] | recs["val"] == set(DS1)
    assert sum(len(v) for v in s.values()) == len(record)


def test_val_must_come_from_ds1() -> None:
    with pytest.raises(ValueError):
        inter_patient_splits(np.array(DS2), [DS2[0]])


def test_intra_split_partitions_beats() -> None:
    y = np.array([0] * 80 + [1] * 10 + [2] * 10)
    s = intra_patient_splits(y, seed=0)
    allidx = np.concatenate(list(s.values()))
    assert len(allidx) == len(y) == len(np.unique(allidx))


def test_process_record_synthetic() -> None:
    rng = np.random.default_rng(0)
    n = 30 * FS
    signal = rng.normal(0, 0.05, n).astype(np.float32)
    samples = np.arange(FS // 2, n - FS // 2, FS)
    signal[samples] += 1.0
    symbols = ["N"] * len(samples)
    symbols[5] = "V"
    symbols[7] = "+"  # rhythm annotation, not a beat
    cfg = {"low_hz": 0.5, "high_hz": 40.0, "order": 3}
    beats, rr, y, r = process_record(signal, samples, symbols, before=90, after=166, filter_cfg=cfg)
    assert beats.shape[1] == 256
    assert len(beats) == len(rr) == len(y) == len(r)
    assert 7 not in [list(samples).index(v) for v in r]  # non-beat annotation dropped
    assert (y == 2).sum() == 1  # one V beat
    assert not np.isnan(rr).any()
