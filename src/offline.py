"""Offline tooling shared by scripts/process_samples.py and the web demo.

One decode pass per video produces: the Part A analysis, a Part B risk curve,
a small proxy video (for rendering) and per-frame brightness (for EDA).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

from .config import COCO_NAMES, PERSON, VEHICLE_CLASSES, pipeline_config
from .pipeline import Analysis, Detection, analyze, detect
from .risk import CausalRisk
from .video import output_size, probe

PROXY_WIDTH = 640
EVENT_COLOURS = {  # BGR
    "accident": (40, 40, 230), "near_miss": (0, 140, 255), "red_light": (0, 0, 200),
    "wrong_way": (200, 0, 200), "illegal_u_turn": (180, 60, 180), "stopped_vehicle": (0, 200, 255),
    "jaywalking": (0, 215, 120), "failure_to_yield": (60, 180, 60), "illegal_turn": (200, 120, 60),
    "solid_line_crossing": (180, 180, 0), "stop_line": (60, 60, 160), "congestion": (130, 130, 130),
    "road_obstacle": (90, 160, 200), "fire_smoke": (30, 30, 30),
}


@dataclass
class Processed:
    analysis: Analysis
    detection: Detection
    risk_t: np.ndarray
    risk: np.ndarray
    brightness: np.ndarray
    proxy: Optional[Path]


def process_video(path: str, work_dir: Path, shared_tracker: bool = False,
                  progress: Optional[Callable[[float, str], None]] = None) -> Processed:
    """Analyse a video once. With `shared_tracker` the risk model reuses Part A's
    (causal) tracks - used by the CPU demo; otherwise it runs its own detector
    exactly as in the submission harness."""
    cfg = pipeline_config()
    meta = probe(path)
    stride = max(1, int(round(meta.fps / cfg.target_fps)))
    work_dir.mkdir(parents=True, exist_ok=True)
    proxy_path = work_dir / (Path(path).stem + "_proxy.mp4")
    pw, ph = output_size(meta, PROXY_WIDTH)
    writer = cv2.VideoWriter(str(proxy_path), cv2.VideoWriter_fourcc(*"mp4v"), meta.fps / stride, (pw, ph))
    risk_model = CausalRisk()
    aw, ah = output_size(meta, cfg.analysis_width)
    risk_model.reset({"fps": meta.fps / stride, "width": aw, "height": ah})
    risk_t, risk, bright = [], [], []

    def on_frame(idx, t, frame, dets):
        risk_t.append(t)
        risk.append(risk_model.observe(dets, t) if shared_tracker else risk_model.step(frame, t))
        small = cv2.resize(frame, (pw, ph), interpolation=cv2.INTER_AREA)
        bright.append(float(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).mean()))
        writer.write(small)

    try:
        a = analyze(path, cfg, progress=progress, on_frame=on_frame)
    finally:
        writer.release()
    return Processed(a, detect(a), np.asarray(risk_t), np.asarray(risk), np.asarray(bright), proxy_path)


# --------------------------------------------------------------- rendering
def _draw_frame(img: np.ndarray, rows: np.ndarray, scale: float, t: float,
                events: list, risk: float) -> np.ndarray:
    for r in rows:
        tid, cls = int(r[2]), int(r[3])
        x1, y1, x2, y2 = (r[5:9] * scale).astype(int)
        colour = (255, 170, 0) if cls in VEHICLE_CLASSES else (0, 165, 255) if cls == PERSON else (200, 200, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, 1)
        cv2.putText(img, f"{COCO_NAMES.get(cls, cls)} {tid}", (x1, max(10, y1 - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, colour, 1, cv2.LINE_AA)
    active = [e for e in events if e.start <= t <= e.end]
    active_ids = {tid for e in active for tid in e.tracks}
    for r in rows:
        if int(r[2]) in active_ids:
            x1, y1, x2, y2 = (r[5:9] * scale).astype(int)
            cv2.rectangle(img, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (0, 0, 255), 2)
    y = 18
    for e in active:
        label = f"{e.label.upper()}  {e.start:.1f}-{e.end:.1f}s"
        cv2.rectangle(img, (4, y - 13), (12 + 8 * len(label), y + 5), EVENT_COLOURS.get(e.label, (0, 0, 0)), -1)
        cv2.putText(img, label, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22
    h, w = img.shape[:2]
    cv2.rectangle(img, (w - 130, 6), (w - 6, 22), (40, 40, 40), -1)
    cv2.rectangle(img, (w - 130, 6), (w - 130 + int(124 * risk), 22),
                  (0, 0, 230) if risk >= 0.5 else (0, 190, 255), -1)
    cv2.putText(img, f"risk {risk:.2f}", (w - 126, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f"t={t:6.1f}s", (w - 90, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def render(p: Processed, out_path: Path) -> Path:
    """Annotated H.264 mp4 (browser-playable) from the proxy video."""
    import imageio_ffmpeg

    cap = cv2.VideoCapture(str(p.proxy))
    fps = cap.get(cv2.CAP_PROP_FPS) or 5.0
    table, times = p.analysis.table, p.analysis.frame_times
    frame_col = table[:, 0].astype(int) if len(table) else np.zeros(0, int)
    frame_ids = np.round(times * p.analysis.meta.fps).astype(int)
    scale = None
    writer = None
    k = 0
    try:
        while k < len(times):
            ok, img = cap.read()
            if not ok:
                break
            if writer is None:
                h, w = img.shape[:2]
                scale = w / p.analysis.width
                writer = imageio_ffmpeg.write_frames(str(out_path), (w, h), fps=fps, codec="libx264",
                                                     pix_fmt_in="bgr24", quality=None, macro_block_size=1,
                                                     output_params=["-crf", "28", "-preset", "veryfast",
                                                                    "-movflags", "+faststart"])
                writer.send(None)
            rows = table[frame_col == frame_ids[k]]
            writer.send(np.ascontiguousarray(_draw_frame(img, rows, scale, times[k], p.detection.events,
                                                         float(p.risk[k]) if k < len(p.risk) else 0.0)))
            k += 1
    finally:
        cap.release()
        if writer is not None:
            writer.close()
    return out_path


# ------------------------------------------------------------ result shapes
def expand_risk(p: Processed) -> List[List[float]]:
    """One [t_sec, score] per original frame, as the harness records it."""
    meta = p.analysis.meta
    t_all = np.arange(meta.n_frames) / meta.fps
    if len(p.risk_t) == 0:
        return [[round(float(t), 3), 0.0] for t in t_all]
    idx = np.clip(np.searchsorted(p.risk_t, t_all, side="right") - 1, 0, len(p.risk) - 1)
    score = np.where(t_all < p.risk_t[0], 0.0, p.risk[idx])
    return [[round(float(t), 3), round(float(s), 4)] for t, s in zip(t_all, score)]


def summary(p: Processed) -> Dict:
    """Compact JSON used by the website."""
    a = p.analysis
    return {
        "video": Path(a.meta.path).name,
        "duration": round(a.meta.duration, 2), "fps": a.meta.fps,
        "resolution": [a.meta.width, a.meta.height],
        "events": [e.as_list() + [e.tracks, e.note] for e in p.detection.events],
        "risk": [[round(float(t), 2), round(float(s), 3)] for t, s in zip(p.risk_t, p.risk)],
        "n_tracks": {COCO_NAMES[c]: int(sum(1 for tr in p.detection.tracks.values() if tr.cls == c))
                     for c in COCO_NAMES},
        "truncated": a.truncated,
    }
