"""Part A pipeline: decode -> detect+track -> scene model -> rules -> segments."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from . import events as ev
from .config import PipelineConfig, RuleConfig, pipeline_config
from .detector import TrackingDetector
from .scene import SceneModel, build_scene
from .tracks import Track, build_tracks, load_table, rows_from_detections, save_table
from .video import VideoMeta, iter_frames, output_size, probe

ProgressFn = Callable[[float, str], None]
FrameHook = Callable[[int, float, np.ndarray, object], None]


@dataclass
class Analysis:
    meta: VideoMeta
    table: np.ndarray          # flat detection table, see tracks.COLS
    frame_times: np.ndarray    # timestamps of analysed frames
    width: int                 # analysis frame size
    height: int
    truncated: bool = False    # stopped early because of the time budget

    @property
    def fps_analysed(self) -> float:
        if len(self.frame_times) < 2:
            return 1.0
        return 1.0 / float(np.median(np.diff(self.frame_times)))


def analyze(path: str, cfg: Optional[PipelineConfig] = None,
            progress: Optional[ProgressFn] = None, on_frame: Optional[FrameHook] = None) -> Analysis:
    """Decode sampled frames, detect + track road users.

    `on_frame(frame_idx, t, frame, detections)` lets offline tools (sample
    processing, the web demo) reuse the same decode pass.
    """
    cfg = cfg or pipeline_config()
    meta = probe(path)
    stride = max(1, int(round(meta.fps / cfg.target_fps)))
    width, height = output_size(meta, cfg.analysis_width)
    det = TrackingDetector(cfg.detector)
    rows: List[list] = []
    times = []
    t_start, truncated = time.time(), False
    # frame count can be missing in some containers: fall back to a 10-minute budget
    budget = cfg.time_budget_ratio * (meta.duration if meta.duration > 1.0 else 600.0)
    for k, (idx, t, frame) in enumerate(iter_frames(meta, stride, width)):
        dets = det.update(frame)
        rows += rows_from_detections(idx, t, dets)
        if on_frame is not None:
            on_frame(idx, t, frame, dets)
        times.append(t)
        if progress and k % 25 == 0:
            progress(min(1.0, t / max(meta.duration, 1e-6)), f"analysed {t:.0f}/{meta.duration:.0f} s")
        if time.time() - t_start > budget:
            truncated = True
            break
    table = np.asarray(rows, np.float64).reshape(-1, 9)
    if times and meta.duration < times[-1]:          # unreliable container metadata
        meta.n_frames = int(round(times[-1] * meta.fps)) + stride
    return Analysis(meta, table, np.asarray(times), width, height, truncated)


def save_analysis(a: Analysis, path: Path) -> None:
    save_table(path, a.table, {"fps": a.meta.fps, "width": a.width, "height": a.height,
                               "src_width": a.meta.width, "src_height": a.meta.height,
                               "n_frames": a.meta.n_frames, "truncated": a.truncated,
                               "video": str(a.meta.path)})
    np.savez_compressed(path.with_suffix(".aux.npz"), frame_times=a.frame_times)


def load_analysis(path: Path) -> Analysis:
    table, m = load_table(path)
    aux = np.load(path.with_suffix(".aux.npz"))
    meta = VideoMeta(str(m["video"]), float(m["fps"]), int(m["src_width"]), int(m["src_height"]),
                     int(m["n_frames"]))
    return Analysis(meta, table, aux["frame_times"], int(m["width"]), int(m["height"]),
                    bool(m["truncated"]))


@dataclass
class Detection:
    events: List[ev.Event]
    tracks: Dict[int, Track]
    scene: SceneModel


def detect(a: Analysis, rules: Optional[RuleConfig] = None) -> Detection:
    rules = rules or RuleConfig()
    tracks = build_tracks(a.table, a.fps_analysed)
    scene = build_scene(tracks, a.width, a.height)
    duration = a.meta.duration
    snap = ev.Snapshot(tracks, duration)
    eps = ev.stationary_episodes(tracks, rules)

    cands: List[ev.Event] = []
    cands += ev.rule_stopped_vehicle(eps, snap, scene, rules, duration)
    cands += ev.rule_congestion(snap, scene, rules)
    cands += ev.rule_wrong_way(tracks, scene, rules)
    cands += ev.rule_jaywalking(tracks, scene, rules)
    cands += ev.rule_accident(tracks, rules)
    red, stopline = ev.rule_red_light(tracks, scene, rules)
    cands += red + stopline
    cands += ev.rule_failure_to_yield(tracks, scene)

    enabled = list(rules.enabled)
    if scene.stop_lines():                  # stop-line geometry configured
        enabled += ["red_light", "stop_line"]
    if scene.has_crossings:
        enabled += ["failure_to_yield"]
    return Detection(ev.finalize(cands, duration, rules, enabled), tracks, scene)


def run(path: str, progress: Optional[ProgressFn] = None) -> List[list]:
    return [e.as_list() for e in detect(analyze(path, progress=progress)).events]
