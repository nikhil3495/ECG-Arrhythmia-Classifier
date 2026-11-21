"""Shared helpers: logging, seeding, config loading, JSON I/O and project paths."""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "mitdb"
PROCESSED_DIR = DATA_DIR / "processed"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
EXPORT_DIR = PROJECT_ROOT / "exports"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"
CONFIG_DIR = PROJECT_ROOT / "configs"

_LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Return a module logger, configuring the root handler once."""
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format=_LOG_FORMAT)
    return logging.getLogger(name)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and (if installed) PyTorch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:  # pragma: no cover - torch is a core dependency
        pass


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config. A top-level ``base`` key is merged in first (shallow per section)."""
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh) or {}
    base = cfg.pop("base", None)
    if base:
        merged = load_config(path.parent / base)
        for key, value in cfg.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = {**merged[key], **value}
            else:
                merged[key] = value
        cfg = merged
    return cfg


def save_json(obj: Any, path: str | Path) -> None:
    """Write ``obj`` as pretty UTF-8 JSON, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=False, default=_json_default)
        fh.write("\n")


def load_json(path: str | Path) -> Any:
    """Read a UTF-8 JSON file."""
    with Path(path).open(encoding="utf-8") as fh:
        return json.load(fh)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serialisable: {type(obj)!r}")
