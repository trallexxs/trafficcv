"""WIUT Hackathon CV track - submission interface.

Part A: detect_events(video_path) -> [[start_sec, end_sec, label], ...]
Part B: RiskEstimator.reset(meta) / step(frame, t_sec) -> P(accident within 5 s)

Implementation lives in src/ (see README.md for the pipeline).
"""
import numpy as np

from src.config import CLASSES  # noqa: F401  (re-exported for the harness)
from src.pipeline import run as _run_part_a


def detect_events(video_path: str) -> list:
    """Part A. Return [[start_sec, end_sec, label], ...] for one .mp4."""
    return _run_part_a(video_path)


class RiskEstimator:
    """Part B. Causal: step() sees frames in order and nothing else."""

    def __init__(self):
        self._impl = None

    def reset(self, meta: dict) -> None:
        if self._impl is None:
            from src.risk import CausalRisk
            self._impl = CausalRisk()
        self._impl.reset(meta)

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        if self._impl is None:
            self.reset({})
        return self._impl.step(frame, t_sec)
