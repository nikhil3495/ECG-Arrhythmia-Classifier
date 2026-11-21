"""On-device ECG arrhythmia classification: 1D-CNN, int8 PTQ, LiteRT deployment."""

__version__ = "1.0.0"

AAMI_CLASSES: tuple[str, ...] = ("N", "S", "V", "F", "Q")
"""AAMI EC57 heartbeat superclasses, in model output order."""

SCORED_CLASSES: tuple[str, ...] = ("N", "S", "V", "F")
"""Classes averaged into the headline macro-F1 (Q has <0.1% of beats once paced records are removed)."""
