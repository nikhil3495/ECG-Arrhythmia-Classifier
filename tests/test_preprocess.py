import numpy as np
import pytest

from ecg_edge import AAMI_CLASSES
from ecg_edge.preprocess import FS, RR_FEATURE_NAMES, bandpass, rr_features, segment_beats, symbol_to_class, znorm


@pytest.mark.parametrize(
    ("symbols", "cls"),
    [
        ("NLRej", "N"),
        ("AaJS", "S"),
        ("VE", "V"),
        ("F", "F"),
        ("/fQ", "Q"),
    ],
)
def test_aami_mapping(symbols: str, cls: str) -> None:
    for s in symbols:
        assert symbol_to_class(s) == AAMI_CLASSES.index(cls)


@pytest.mark.parametrize("symbol", ["+", "~", "|", '"', "x", "!", "[", "]"])
def test_non_beat_symbols_are_ignored(symbol: str) -> None:
    assert symbol_to_class(symbol) is None


def test_segmentation_shapes_and_edges() -> None:
    signal = np.arange(1000, dtype=np.float32)
    r = np.array([10, 100, 500, 900, 990])
    beats, valid = segment_beats(signal, r, before=90, after=166)
    assert valid.tolist() == [False, True, True, False, False]
    assert beats.shape == (2, 256)
    # window is centred so that index `before` is the R-peak sample
    assert beats[0, 90] == 100
    assert beats[1, 0] == 500 - 90


def test_znorm() -> None:
    rng = np.random.default_rng(0)
    b = znorm(rng.normal(3, 5, size=(4, 256)))
    np.testing.assert_allclose(b.mean(1), 0, atol=1e-5)
    np.testing.assert_allclose(b.std(1), 1, atol=1e-3)


def test_rr_features_regular_rhythm() -> None:
    r = np.arange(0, 20 * FS, FS)  # 60 bpm
    f = rr_features(r)
    assert f.shape == (len(r), len(RR_FEATURE_NAMES))
    assert np.isnan(f[0]).all() and np.isnan(f[-1]).all()
    np.testing.assert_allclose(f[1:-1, :4], 1.0, rtol=1e-6)  # all intervals 1 s
    np.testing.assert_allclose(f[1:-1, 4:], 1.0, rtol=1e-6)  # ratios 1


def test_rr_features_premature_beat() -> None:
    r = np.array([0, 360, 720, 900, 1440, 1800])  # beat 3 comes early (0.5 s), compensatory pause
    f = rr_features(r)
    pre, post = f[3, 0], f[3, 1]
    assert pre == pytest.approx(0.5)
    assert post == pytest.approx(1.5)
    assert f[3, 4] < 1  # pre / local < 1 flags prematurity
    assert f[3, 6] < 1  # pre / long < 1 as well


def test_bandpass_removes_baseline() -> None:
    t = np.arange(10 * FS) / FS
    sig = 5.0 + 2.0 * t + np.sin(2 * np.pi * 10 * t)
    out = bandpass(sig)
    assert abs(out[FS:-FS].mean()) < 0.05
    assert out.dtype == np.float32
