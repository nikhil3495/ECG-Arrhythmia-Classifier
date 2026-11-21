"""Quantisation-friendly 1D-CNN with an RR-interval side branch.

Only Conv1d, BatchNorm1d (folded at export), ReLU, MaxPool1d, mean-pooling, Linear and
concatenation are used - all map to int8 LiteRT builtins.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from ecg_edge import AAMI_CLASSES
from ecg_edge.preprocess import RR_FEATURE_NAMES


class ConvBlock(nn.Sequential):
    """Conv1d -> BatchNorm1d -> ReLU -> MaxPool1d(2)."""

    def __init__(self, c_in: int, c_out: int, kernel_size: int) -> None:
        super().__init__(
            nn.Conv1d(c_in, c_out, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(c_out),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )


class ECGNet(nn.Module):
    """Beat-morphology CNN + RR-feature MLP, fused by concatenation.

    Inputs:
        beat: ``(B, 1, 256)`` z-normalised beat.
        rr: ``(B, n_rr)`` raw RR features (seconds / ratios). They are standardised inside the model with
            training-set statistics stored as buffers, so deployed models take raw features.

    Output:
        ``(B, 5)`` logits over the AAMI classes N, S, V, F, Q.
    """

    def __init__(
        self,
        widths: Sequence[int] = (16, 32, 32, 64),
        kernel_size: int = 7,
        rr_hidden: int = 16,
        fc_hidden: int = 32,
        dropout: float = 0.2,
        n_rr: int = len(RR_FEATURE_NAMES),
        n_classes: int = len(AAMI_CLASSES),
    ) -> None:
        super().__init__()
        chans = [1, *widths]
        self.features = nn.Sequential(*[ConvBlock(chans[i], chans[i + 1], kernel_size) for i in range(len(widths))])
        self.register_buffer("rr_mean", torch.zeros(n_rr))
        self.register_buffer("rr_std", torch.ones(n_rr))
        self.rr_folded = False  # set by fold_rr_normalization() for export
        self.rr_branch = nn.Sequential(nn.Linear(n_rr, rr_hidden), nn.ReLU(inplace=True))
        self.head = nn.Sequential(
            nn.Linear(widths[-1] + rr_hidden, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, n_classes),
        )

    def set_rr_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Store training-set RR feature statistics used for in-model standardisation."""
        self.rr_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.rr_std.copy_(torch.as_tensor(std, dtype=torch.float32).clamp_min(1e-6))

    def forward(self, beat: torch.Tensor, rr: torch.Tensor) -> torch.Tensor:  # noqa: D102
        h = self.features(beat).mean(dim=-1)  # global average pooling
        if not self.rr_folded:
            rr = (rr - self.rr_mean) / self.rr_std
        r = self.rr_branch(rr)
        return self.head(torch.cat([h, r], dim=1))


def fold_rr_normalization(model: ECGNet) -> ECGNet:
    """Return an eval-mode copy with RR standardisation folded into the first RR ``Linear``.

    ``W ((rr - mu) / sigma) + b == (W / sigma) rr + (b - W mu / sigma)``, so the exported graph has no
    float ``SUB``/``MUL`` in front of the RR branch and the int8 model is integer-only end to end.
    """
    folded = copy.deepcopy(model).eval()
    lin = folded.rr_branch[0]
    assert isinstance(lin, nn.Linear)
    with torch.no_grad():
        inv = 1.0 / folded.rr_std
        new_bias = lin.bias - lin.weight @ (folded.rr_mean * inv)
        lin.weight.mul_(inv[None, :])
        lin.bias.copy_(new_bias)
        folded.rr_mean.zero_()
        folded.rr_std.fill_(1.0)
    folded.rr_folded = True
    return folded


def build_model(model_cfg: dict[str, Any]) -> ECGNet:
    """Instantiate :class:`ECGNet` from the ``model`` config section."""
    return ECGNet(
        widths=model_cfg["widths"],
        kernel_size=model_cfg.get("kernel_size", 7),
        rr_hidden=model_cfg.get("rr_hidden", 16),
        fc_hidden=model_cfg.get("fc_hidden", 32),
        dropout=model_cfg.get("dropout", 0.2),
    )


def count_params(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_checkpoint(path: str, map_location: str = "cpu") -> tuple[ECGNet, dict[str, Any]]:
    """Load a checkpoint written by :mod:`ecg_edge.train`; returns ``(model.eval(), checkpoint)``."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model = build_model(ckpt["config"]["model"])
    model.load_state_dict(ckpt["state_dict"])
    return model.eval(), ckpt
