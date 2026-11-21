import pytest
import torch
from torch import nn

from ecg_edge.models import ECGNet, build_model, count_params
from ecg_edge.preprocess import RR_FEATURE_NAMES
from ecg_edge.utils import load_config

N_RR = len(RR_FEATURE_NAMES)
ALLOWED = (nn.Conv1d, nn.BatchNorm1d, nn.ReLU, nn.MaxPool1d, nn.Linear, nn.Dropout, nn.Sequential, ECGNet)


@pytest.mark.parametrize("variant", ["tiny", "small", "base"])
def test_forward_shape(variant: str) -> None:
    model = build_model(load_config(f"configs/{variant}.yaml")["model"]).eval()
    out = model(torch.randn(4, 1, 256), torch.rand(4, N_RR) + 0.5)
    assert out.shape == (4, 5)
    assert torch.isfinite(out).all()


def test_variants_increase_in_size() -> None:
    sizes = [count_params(build_model(load_config(f"configs/{v}.yaml")["model"])) for v in ("tiny", "small", "base")]
    assert sizes == sorted(sizes) and len(set(sizes)) == 3


def test_only_quantization_friendly_modules() -> None:
    model = build_model(load_config("configs/small.yaml")["model"])
    for m in model.modules():
        assert isinstance(m, ALLOWED), type(m)


def test_rr_stats_are_buffers_in_state_dict() -> None:
    model = ECGNet()
    model.set_rr_stats(torch.full((N_RR,), 2.0), torch.full((N_RR,), 0.5))
    sd = model.state_dict()
    assert torch.equal(sd["rr_mean"], torch.full((N_RR,), 2.0))
    assert torch.equal(sd["rr_std"], torch.full((N_RR,), 0.5))
