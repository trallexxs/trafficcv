"""Part B: causal accident-risk estimator.

Runs its own detector+tracker on every ~0.2 s of the stream (it never sees
future frames and never uses Part A output) and combines three signals:
  * closest-point-of-approach between pairs of road users under a
    constant-velocity model (a time-to-collision generalisation);
  * hard braking of a tracked vehicle;
  * vehicles moving against the learned traffic direction (scene prior only).
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Deque, Dict, Tuple

import cv2
import numpy as np

from .config import PERSON, VEHICLE_CLASSES, DetectorConfig, RiskConfig, has_cuda, pipeline_config
from .detector import TrackingDetector
from .scene import PRIOR_PATH, SceneModel, SceneStats, load_geometry


class CausalRisk:
    def __init__(self, cfg: RiskConfig | None = None):
        self.cfg = cfg or RiskConfig()
        if cfg is None and not has_cuda():
            self.cfg.stride_s = 0.5
        pcfg = pipeline_config()
        det_cfg = DetectorConfig(weights=pcfg.detector.weights, imgsz=min(self.cfg.imgsz, pcfg.detector.imgsz),
                                 conf=pcfg.detector.conf, iou=pcfg.detector.iou, tracker=pcfg.detector.tracker)
        self.detector = TrackingDetector(det_cfg)
        self.width = pcfg.analysis_width
        self.prior = SceneStats.load(PRIOR_PATH)
        self.reset({"fps": 25.0, "width": 1920, "height": 1080})

    def reset(self, meta: dict) -> None:
        self.detector.reset()
        fps = float(meta.get("fps") or 25.0)
        self.stride = max(1, int(round(self.cfg.stride_s * fps)))
        w, h = int(meta.get("width") or 1920), int(meta.get("height") or 1080)
        self.out_w = min(self.width, w)
        self.out_h = int(round(h * self.out_w / w))
        self.scene = (SceneModel(self.prior, self.out_w, self.out_h, load_geometry())
                      if self.prior is not None else None)
        self.hist: Dict[int, Deque[Tuple[float, np.ndarray, float, int]]] = defaultdict(lambda: deque(maxlen=15))
        self.frame_no = 0
        self.score = 0.0

    # --------------------------------------------------------------- signals
    def _velocity(self, h) -> Tuple[np.ndarray, float] | None:
        if len(h) < 3:
            return None
        t0, p0, _, _ = h[max(0, len(h) - 4)]
        t1, p1, s1, _ = h[-1]
        if t1 - t0 < 1e-3:
            return None
        return (p1 - p0) / (t1 - t0), s1

    def _pair_risk(self, objs) -> float:
        best = 0.0
        for a in range(len(objs)):
            pa, va, sa, ca = objs[a]
            for b in range(a + 1, len(objs)):
                pb, vb, sb, cb = objs[b]
                if ca == PERSON and cb == PERSON:
                    continue
                s = 0.5 * (sa + sb)
                d = pb - pa
                if np.linalg.norm(d) > 6 * s:
                    continue
                dv = vb - va
                closing = -(d @ dv) / (np.linalg.norm(d) + 1e-6) / s    # body lengths / s
                if closing < 0.6:
                    continue
                tstar = -(d @ dv) / (dv @ dv + 1e-9)
                if not 0.0 < tstar < self.cfg.horizon_s:
                    continue
                dmin = np.linalg.norm(d + dv * tstar) / s
                r = np.exp(-(dmin / 0.5) ** 2) * (1.0 / (1.0 + np.exp((tstar - 1.5) / 0.35)))
                best = max(best, float(r * min(1.0, closing / 1.5)))
        return best

    def _braking(self, h) -> float:
        if len(h) < 8:
            return 0.0
        pts = np.array([p for _, p, _, _ in h])
        ts = np.array([t for t, _, _, _ in h])
        s = h[-1][2]
        v_old = np.linalg.norm(pts[-5] - pts[-8]) / max(ts[-5] - ts[-8], 1e-3) / s
        v_new = np.linalg.norm(pts[-1] - pts[-3]) / max(ts[-1] - ts[-3], 1e-3) / s
        return 0.45 if v_old > 1.2 and v_new < 0.3 * v_old else 0.0

    def _update(self, frame: np.ndarray, t: float) -> float:
        if frame.shape[1] != self.out_w:
            frame = cv2.resize(frame, (self.out_w, self.out_h), interpolation=cv2.INTER_AREA)
        return self.update_from_detections(self.detector.update(frame), t)

    def update_from_detections(self, dets, t: float) -> float:
        """Risk from one frame of tracked detections (analysis-frame pixels).

        Used directly by the web demo, which shares Part A's causal tracker to
        halve CPU time; the submission harness always goes through `step`.
        """
        objs, brake, wrong = [], 0.0, 0.0
        seen = set()
        for tid, c, box in zip(dets.ids, dets.cls, dets.boxes):
            if tid < 0 or (c not in VEHICLE_CLASSES and c != PERSON):
                continue
            foot = np.array([(box[0] + box[2]) / 2, box[3]])
            size = float(np.sqrt(max((box[2] - box[0]) * (box[3] - box[1]), 1.0)))
            h = self.hist[tid]
            h.append((t, foot, size, int(c)))
            seen.add(tid)
            vel = self._velocity(h)
            if vel is None:
                continue
            objs.append((foot, vel[0], vel[1], int(c)))
            if c in VEHICLE_CLASSES:
                brake = max(brake, self._braking(h))
                if self.scene is not None and np.linalg.norm(vel[0]) / size > 0.8 and \
                        self.scene.against_flow(foot, vel[0], 0.65):
                    wrong = 0.3
        for tid in [k for k, h in self.hist.items() if k not in seen and t - h[-1][0] > 3.0]:
            del self.hist[tid]
        parts = [self._pair_risk(objs), brake, wrong]
        return float(1.0 - np.prod([1.0 - p for p in parts]))

    def _smooth(self, raw: float) -> float:
        # fast attack, slower decay: an alarm stays up briefly after a peak
        self.score = raw if raw > self.score else self.cfg.ema * self.score + (1 - self.cfg.ema) * raw
        return float(np.clip(self.score, 0.0, 1.0))

    def observe(self, dets, t: float) -> float:
        """Smoothed risk from precomputed detections (web demo path)."""
        return self._smooth(self.update_from_detections(dets, t))

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        if self.frame_no % self.stride == 0:
            self._smooth(self._update(frame, t_sec))
        self.frame_no += 1
        return float(np.clip(self.score, 0.0, 1.0))
