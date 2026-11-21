"""Minimal LiteRT (.tflite) inference wrapper - the only runtime the deployed demo needs.

Handles both float models and full-integer int8 models (whose inputs/outputs are int8 and must be
(de)quantised with the scale/zero-point stored in the flatbuffer).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _interpreter_cls() -> Any:
    """Return the LiteRT interpreter class (ai-edge-litert, falling back to tflite_runtime / TF)."""
    try:
        from ai_edge_litert.interpreter import Interpreter

        return Interpreter
    except ImportError:  # pragma: no cover - fallbacks for other environments
        try:
            from tflite_runtime.interpreter import Interpreter  # type: ignore[import-not-found]

            return Interpreter
        except ImportError:
            import tensorflow as tf  # type: ignore[import-not-found]

            return tf.lite.Interpreter


class LiteRTClassifier:
    """Single-beat classifier backed by a LiteRT interpreter.

    The model has two inputs - beat ``(1, 1, 256)`` and RR features ``(1, n_rr)`` - identified by rank,
    and one ``(1, 5)`` logits output.
    """

    def __init__(self, model_path: str | Path, num_threads: int = 1) -> None:
        self.model_path = Path(model_path)
        self.interpreter = _interpreter_cls()(model_path=str(model_path), num_threads=num_threads)
        self.interpreter.allocate_tensors()
        inputs = self.interpreter.get_input_details()
        self._beat = next(d for d in inputs if len(d["shape"]) == 3)
        self._rr = next(d for d in inputs if len(d["shape"]) == 2)
        self._out = self.interpreter.get_output_details()[0]

    @property
    def input_dtype(self) -> np.dtype:
        """Dtype of the beat input tensor (``int8`` for full-integer models)."""
        return np.dtype(self._beat["dtype"])

    @staticmethod
    def _quantize(arr: np.ndarray, detail: dict[str, Any]) -> np.ndarray:
        dtype = np.dtype(detail["dtype"])
        if dtype == np.float32:
            return arr.astype(np.float32)
        scale, zero_point = detail["quantization"]
        info = np.iinfo(dtype)
        q = np.round(arr / scale) + zero_point
        return np.clip(q, info.min, info.max).astype(dtype)

    def _dequantize(self, arr: np.ndarray) -> np.ndarray:
        if np.dtype(self._out["dtype"]) == np.float32:
            return arr.astype(np.float32)
        scale, zero_point = self._out["quantization"]
        return (arr.astype(np.float32) - zero_point) * scale

    def set_inputs(self, beat: np.ndarray, rr: np.ndarray) -> None:
        """Quantise (if needed) and copy one beat ``(1,1,256)`` and its RR features ``(1,n_rr)`` in."""
        self.interpreter.set_tensor(self._beat["index"], self._quantize(beat.reshape(self._beat["shape"]), self._beat))
        self.interpreter.set_tensor(self._rr["index"], self._quantize(rr.reshape(self._rr["shape"]), self._rr))

    def invoke(self) -> None:
        """Run the graph on the inputs already set (used for latency measurement)."""
        self.interpreter.invoke()

    def output_logits(self) -> np.ndarray:
        """Dequantised logits of the last invocation, shape ``(5,)``."""
        return self._dequantize(self.interpreter.get_tensor(self._out["index"]))[0]

    def predict_logits(self, beats: np.ndarray, rr: np.ndarray) -> np.ndarray:
        """Logits for a batch of beats, run one at a time: ``(n,1,256), (n,n_rr) -> (n,5)``."""
        out = np.empty((len(beats), int(self._out["shape"][-1])), dtype=np.float32)
        for i in range(len(beats)):
            self.set_inputs(beats[i : i + 1], rr[i : i + 1])
            self.invoke()
            out[i] = self.output_logits()
        return out


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis."""
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)
