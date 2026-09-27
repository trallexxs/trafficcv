"""Exploratory data analysis of processed videos -> images + JSON for the website."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from .config import COCO_NAMES, PERSON
from .offline import Processed
from .scene import GX, GY, NBINS, SceneStats


def background(proxy: Path, n: int = 40) -> np.ndarray:
    """Median of frames spread over the video = the empty scene."""
    cap = cv2.VideoCapture(str(proxy))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    frames = []
    for k in np.linspace(0, total - 1, min(n, total)).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(k))
        ok, f = cap.read()
        if ok:
            frames.append(f)
    cap.release()
    return np.median(np.stack(frames), 0).astype(np.uint8) if frames else np.zeros((360, 640, 3), np.uint8)


def _heat_overlay(bg: np.ndarray, grid: np.ndarray, cmap=cv2.COLORMAP_JET) -> np.ndarray:
    g = np.log1p(grid.astype(np.float32))
    g = (255 * g / max(g.max(), 1e-6)).astype(np.uint8)
    g = cv2.resize(g, (bg.shape[1], bg.shape[0]), interpolation=cv2.INTER_CUBIC)
    col = cv2.applyColorMap(g, cmap)
    alpha = (g[..., None] / 255.0) * 0.65
    return (bg * (1 - alpha) + col * alpha).astype(np.uint8)


def trajectories_image(bg: np.ndarray, p: Processed, max_tracks: int = 400) -> np.ndarray:
    img = (bg * 0.6).astype(np.uint8)
    scale = bg.shape[1] / p.analysis.width
    tracks = sorted(p.detection.tracks.values(), key=lambda tr: -tr.duration)[:max_tracks]
    for tr in tracks:
        pts = (tr.foot * scale).astype(np.int32)
        if len(pts) < 2:
            continue
        d = pts[-1] - pts[0]
        hue = int((np.degrees(np.arctan2(d[1], d[0])) % 360) / 2)
        colour = (0, 165, 255) if tr.cls == PERSON else tuple(
            int(c) for c in cv2.cvtColor(np.uint8([[[hue, 230, 255]]]), cv2.COLOR_HSV2BGR)[0, 0])
        cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, colour, 1, cv2.LINE_AA)
    return img


def direction_field_image(bg: np.ndarray, stats: SceneStats) -> np.ndarray:
    img = (bg * 0.55).astype(np.uint8)
    h, w = img.shape[:2]
    cw, ch = w / GX, h / GY
    tot = stats.dirs.sum(2)
    thr = max(5.0, np.percentile(tot[tot > 0], 50)) if (tot > 0).any() else np.inf
    for gy in range(GY):
        for gx in range(GX):
            if tot[gy, gx] < thr:
                continue
            b = int(stats.dirs[gy, gx].argmax())
            dom = stats.dirs[gy, gx, b] / tot[gy, gx]
            ang = (b + 0.5) / NBINS * 2 * np.pi - np.pi
            c = np.array([(gx + 0.5) * cw, (gy + 0.5) * ch])
            e = c + 0.45 * min(cw, ch) * 2 * np.array([np.cos(ang), np.sin(ang)])
            colour = (0, 255, 0) if dom > 0.65 else (0, 200, 255)
            cv2.arrowedLine(img, tuple(c.astype(int)), tuple(e.astype(int)), colour, 1, cv2.LINE_AA, tipLength=0.4)
    return img


def counts_over_time(p: Processed, bin_s: float = 10.0) -> Dict[str, List]:
    dur = max(p.analysis.meta.duration, bin_s)
    edges = np.arange(0, dur + bin_s, bin_s)
    out = {"t": edges[:-1].round(1).tolist()}
    for c, name in COCO_NAMES.items():
        firsts = [tr.t[0] for tr in p.detection.tracks.values() if tr.cls == c]
        out[name] = np.histogram(firsts, edges)[0].tolist() if firsts else [0] * (len(edges) - 1)
    t = p.analysis.table
    per_frame = {}
    for fi in np.unique(t[:, 0]) if len(t) else []:
        per_frame[fi] = int((t[:, 0] == fi).sum())
    occ_t = np.array(sorted(per_frame)) / p.analysis.meta.fps if per_frame else np.zeros(0)
    occ = np.array([per_frame[k] for k in sorted(per_frame)]) if per_frame else np.zeros(0)
    out["objects_in_view"] = [np.round(occ_t[::5], 1).tolist(), occ[::5].tolist()]
    return out


def write_eda(items: List[Processed], stats: SceneStats, out_dir: Path) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    bg = background(items[0].proxy)
    cv2.imwrite(str(out_dir / "background.jpg"), bg)
    cv2.imwrite(str(out_dir / "heat_vehicles.jpg"), _heat_overlay(bg, stats.veh + stats.stat))
    cv2.imwrite(str(out_dir / "heat_pedestrians.jpg"), _heat_overlay(bg, stats.ped, cv2.COLORMAP_HOT))
    cv2.imwrite(str(out_dir / "heat_stationary.jpg"), _heat_overlay(bg, stats.stat, cv2.COLORMAP_OCEAN))
    cv2.imwrite(str(out_dir / "direction_field.jpg"), direction_field_image(bg, stats))
    cv2.imwrite(str(out_dir / "trajectories.jpg"), trajectories_image(bg, items[0]))
    speeds = np.concatenate([tr.speed[tr.speed > 0.3] for p in items for tr in p.detection.tracks.values()
                             if tr.is_vehicle] or [np.zeros(0)])
    report = {
        "videos": [{
            "video": Path(p.analysis.meta.path).name,
            "resolution": f"{p.analysis.meta.width}x{p.analysis.meta.height}",
            "fps": round(p.analysis.meta.fps, 2),
            "duration_s": round(p.analysis.meta.duration, 1),
            "frames": p.analysis.meta.n_frames,
            "brightness_mean": round(float(p.brightness.mean()), 1) if len(p.brightness) else None,
            "brightness_curve": [np.round(p.risk_t[::10], 1).tolist(), np.round(p.brightness[::10], 1).tolist()],
            "tracks": {COCO_NAMES[c]: int(sum(1 for tr in p.detection.tracks.values() if tr.cls == c))
                       for c in COCO_NAMES},
            "counts": counts_over_time(p),
        } for p in items],
        "vehicle_speed_hist": np.histogram(speeds, bins=30, range=(0, 6))[0].tolist(),
        "speed_bins": np.linspace(0, 6, 31).round(2).tolist(),
    }
    with open(out_dir / "eda.json", "w") as f:
        json.dump(report, f)
    return report
