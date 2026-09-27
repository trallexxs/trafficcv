"""YOLO detector + ByteTrack tracker (both online / causal)."""
from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np

from .config import DETECT_CLASSES, SEED, DetectorConfig

os.environ.setdefault("YOLO_OFFLINE", "1")
os.environ.setdefault("YOLO_VERBOSE", "False")


def set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except Exception:
        pass


@dataclass
class FrameDetections:
    ids: np.ndarray     # (N,) int, -1 when not yet confirmed by the tracker
    cls: np.ndarray     # (N,) int COCO id
    conf: np.ndarray    # (N,) float
    boxes: np.ndarray   # (N, 4) float xyxy in analysis-frame pixels


class TrackingDetector:
    """Stateful wrapper: call `update(frame)` on consecutive sampled frames."""

    def __init__(self, cfg: DetectorConfig):
        import torch
        from ultralytics import YOLO

        set_seeds()
        self.cfg = cfg
        self.model = YOLO(cfg.weights, task="detect")
        self.cuda = torch.cuda.is_available()
        self.device = 0 if self.cuda else "cpu"

    def reset(self) -> None:
        predictor = getattr(self.model, "predictor", None)
        if predictor is not None and hasattr(predictor, "trackers"):
            for tracker in predictor.trackers:
                tracker.reset()

    def update(self, frame: np.ndarray) -> FrameDetections:
        res = self.model.track(
            frame, persist=True, tracker=self.cfg.tracker, imgsz=self.cfg.imgsz,
            conf=self.cfg.conf, iou=self.cfg.iou, classes=DETECT_CLASSES,
            device=self.device, half=self.cuda, verbose=False)[0]
        b = res.boxes
        if b is None or len(b) == 0:
            return FrameDetections(np.zeros(0, int), np.zeros(0, int), np.zeros(0), np.zeros((0, 4)))
        ids = b.id.cpu().numpy().astype(int) if b.id is not None else -np.ones(len(b), int)
        return FrameDetections(ids, b.cls.cpu().numpy().astype(int),
                               b.conf.cpu().numpy().astype(float), b.xyxy.cpu().numpy().astype(float))

