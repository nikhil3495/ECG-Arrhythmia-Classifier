"""LiteRT export parity on a tiny random model and synthetic batch (no dataset needed)."""

from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("litert_torch")

from ecg_edge.data import BeatDataset
from ecg_edge.evaluate import predict_torch
from ecg_edge.models import build_model
from ecg_edge.preprocess import RR_FEATURE_NAMES
from ecg_edge.quantize_export import export_fp32, export_int8
from ecg_edge.runtime import LiteRTClassifier, softmax
from ecg_edge.utils import load_config


@pytest.fixture(scope="module")
def model_and_data() -> tuple[torch.nn.Module, BeatDataset]:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = build_model(load_config("configs/tiny.yaml")["model"])
    n = 64
    rr = (rng.uniform(0.4, 1.4, size=(n, len(RR_FEATURE_NAMES)))).astype(np.float32)
    model.set_rr_stats(torch.from_numpy(rr.mean(0)), torch.from_numpy(rr.std(0)))
    # give BatchNorm non-trivial running stats
    model.train()
    with torch.no_grad():
        for _ in range(5):
            model(torch.randn(32, 1, 256), torch.from_numpy(rr[:32]))
    model.eval()
    x = rng.normal(size=(n, 1, 256)).astype(np.float32)
    ds = BeatDataset(x, rr, np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.int64))
    return model, ds


def test_fp32_tflite_matches_pytorch(model_and_data, tmp_path: Path) -> None:
    model, ds = model_and_data
    path = export_fp32(model, tmp_path / "m_fp32.tflite")
    clf = LiteRTClassifier(path)
    ref = predict_torch(model, ds.x, ds.rr)
    out = clf.predict_logits(ds.x, ds.rr)
    np.testing.assert_allclose(out, ref, atol=1e-4)
    assert (out.argmax(1) == ref.argmax(1)).all()


def test_int8_tflite_is_full_integer_and_close(model_and_data, tmp_path: Path) -> None:
    model, ds = model_and_data
    path = export_int8(model, ds, tmp_path / "m_int8.tflite")
    clf = LiteRTClassifier(path)
    assert clf.input_dtype == np.int8
    ref = softmax(predict_torch(model, ds.x, ds.rr))
    out = softmax(clf.predict_logits(ds.x, ds.rr))
    assert np.abs(out - ref).mean() < 0.05
    fp32 = export_fp32(model, tmp_path / "ref_fp32.tflite")
    assert path.stat().st_size < fp32.stat().st_size
