"""Track table storage and per-track kinematics."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

from .config import PERSON, VEHICLE_CLASSES

# Columns of the flat detection table.
COLS = ["frame", "t", "id", "cls", "conf", "x1", "y1", "x2", "y2"]


@dataclass
class Track:
    tid: int
    cls: int                 # majority COCO class
    t: np.ndarray            # (T,)
    box: np.ndarray          # (T, 4) xyxy
    foot: np.ndarray         # (T, 2) smoothed bottom-centre point (ground contact)
    size: np.ndarray         # (T,) smoothed sqrt(w*h)
    vel: np.ndarray          # (T, 2) px/s (centred difference, smoothed)
    speed: np.ndarray        # (T,) body lengths / s

    @property
    def is_vehicle(self) -> bool:
        return self.cls in VEHICLE_CLASSES

    @property
    def is_person(self) -> bool:
        return self.cls == PERSON

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0]) if len(self.t) > 1 else 0.0

    def index_at(self, t: float) -> int:
        return int(np.clip(np.searchsorted(self.t, t), 0, len(self.t) - 1))


def _smooth(x: np.ndarray, k: int) -> np.ndarray:
    if len(x) < 3 or k <= 1:
        return x.copy()
    k = min(k, len(x) if len(x) % 2 else len(x) - 1)
    pad = k // 2
    xp = np.pad(x, [(pad, pad)] + [(0, 0)] * (x.ndim - 1), mode="edge")
    kernel = np.ones(k) / k
    if x.ndim == 1:
        return np.convolve(xp, kernel, mode="valid")
    return np.stack([np.convolve(xp[:, j], kernel, mode="valid") for j in range(x.shape[1])], 1)


def build_tracks(table: np.ndarray, fps_analysed: float, min_len: int = 3) -> Dict[int, Track]:
    """Turn the flat detection table into Track objects with kinematics."""
    tracks: Dict[int, Track] = {}
    if len(table) == 0:
        return tracks
    table = table[table[:, 2] >= 0]
    win = max(3, int(round(fps_analysed * 0.8)) | 1)      # ~0.8 s smoothing window
    for tid in np.unique(table[:, 2]).astype(int):
        rows = table[table[:, 2] == tid]
        rows = rows[np.argsort(rows[:, 1])]
        if len(rows) < min_len:
            continue
        cls_vals, counts = np.unique(rows[:, 3].astype(int), return_counts=True)
        cls = int(cls_vals[np.argmax(counts)])
        t = rows[:, 1]
        box = rows[:, 5:9]
        foot = np.stack([(box[:, 0] + box[:, 2]) / 2, box[:, 3]], 1)
        foot = _smooth(foot, win)
        size = np.sqrt(np.maximum((box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1]), 1.0))
        size = _smooth(size, win)
        vel = np.gradient(foot, t, axis=0) if len(t) > 1 else np.zeros_like(foot)
        vel = _smooth(vel, win)
        speed = np.linalg.norm(vel, axis=1) / np.maximum(size, 1.0)
        tracks[tid] = Track(tid, cls, t, box, foot, size, vel, speed)
    return tracks


def save_table(path: Path, table: np.ndarray, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, table=table.astype(np.float32), **{k: np.asarray(v) for k, v in meta.items()})


def load_table(path: Path):
    data = np.load(path, allow_pickle=False)
    meta = {k: data[k].item() for k in data.files if k != "table"}
    return data["table"].astype(np.float64), meta


def rows_from_detections(frame_idx: int, t: float, dets) -> List[List[float]]:
    return [[frame_idx, t, int(i), int(c), float(p), *map(float, b)]
            for i, c, p, b in zip(dets.ids, dets.cls, dets.conf, dets.boxes)]
