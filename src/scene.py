"""Scene model of the fixed camera.

Two sources are combined:
  1. configs/scene.json (optional, hand-written from camera.md / a still frame):
     crossings, stop lines, signal position, road polygons - normalised [0,1] coords.
  2. Statistics learned from trajectories: where vehicles drive (road mask),
     which way they drive in every grid cell (direction field) and where
     pedestrians cross (crossing zones). A prior computed on the sample videos
     (weights/scene_prior.npz) is merged with statistics of the video itself.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from .config import CONFIG_DIR, WEIGHTS_DIR
from .tracks import Track

GX, GY, NBINS = 48, 27, 8
PRIOR_PATH = WEIGHTS_DIR / "scene_prior.npz"
SCENE_JSON = CONFIG_DIR / "scene.json"


def _box_blur(a: np.ndarray) -> np.ndarray:
    return cv2.blur(a.astype(np.float32), (3, 3), borderType=cv2.BORDER_REPLICATE)


class SceneStats:
    """Additive trajectory statistics on a GY x GX grid of the normalised frame."""

    def __init__(self):
        self.veh = np.zeros((GY, GX), np.float32)          # moving-vehicle samples
        self.dirs = np.zeros((GY, GX, NBINS), np.float32)  # heading histogram (moving vehicles)
        self.stat = np.zeros((GY, GX), np.float32)         # stationary-vehicle samples
        self.ped = np.zeros((GY, GX), np.float32)          # distinct pedestrian tracks

    def add_tracks(self, tracks: Dict[int, Track], width: int, height: int) -> None:
        for tr in tracks.values():
            gx = np.clip((tr.foot[:, 0] / width * GX).astype(int), 0, GX - 1)
            gy = np.clip((tr.foot[:, 1] / height * GY).astype(int), 0, GY - 1)
            if tr.is_vehicle:
                moving = tr.speed > 0.5
                np.add.at(self.veh, (gy[moving], gx[moving]), 1)
                np.add.at(self.stat, (gy[~moving], gx[~moving]), 1)
                ang = np.arctan2(tr.vel[:, 1], tr.vel[:, 0])
                b = ((ang + np.pi) / (2 * np.pi) * NBINS).astype(int) % NBINS
                np.add.at(self.dirs, (gy[moving], gx[moving], b[moving]), 1)
            elif tr.is_person:
                cells = np.unique(gy * GX + gx)
                np.add.at(self.ped, (cells // GX, cells % GX), 1)

    def merged(self, other: "SceneStats", w_other: float = 1.0) -> "SceneStats":
        out = SceneStats()
        for name in ("veh", "dirs", "stat", "ped"):
            setattr(out, name, getattr(self, name) + w_other * getattr(other, name))
        return out

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, veh=self.veh, dirs=self.dirs, stat=self.stat, ped=self.ped)

    @classmethod
    def load(cls, path: Path) -> Optional["SceneStats"]:
        if not path.exists():
            return None
        d = np.load(path)
        s = cls()
        s.veh, s.dirs, s.stat, s.ped = d["veh"], d["dirs"], d["stat"], d["ped"]
        return s


class SceneModel:
    def __init__(self, stats: SceneStats, width: int, height: int, geometry: Optional[dict] = None):
        self.w, self.h = width, height
        self.geometry = geometry or {}
        veh = stats.veh.astype(np.float32)
        thr = max(3.0, 0.03 * float(np.percentile(veh[veh > 0], 95))) if (veh > 0).any() else np.inf
        kernel = np.ones((3, 3), np.uint8)
        core = cv2.morphologyEx((veh >= 2 * thr).astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        road = cv2.morphologyEx((veh >= thr).astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        self.road_core = core.astype(bool)          # cells vehicles really drive through
        self.road = road.astype(bool) | self.road_core
        dirs = np.stack([_box_blur(stats.dirs[:, :, k]) for k in range(NBINS)], 2)
        total = dirs.sum(2) + 1e-6
        self.dir_share = dirs / total[:, :, None]
        self.dir_support = total
        self.dominant = self.dir_share.argmax(2)
        self.dominance = self.dir_share.max(2)
        self.polys = {k: [self._poly(p) for p in self.geometry.get(k, [])]
                      for k in ("crossings", "road")}
        ped = stats.ped * self.road
        self.crossing_learned = (ped >= max(4.0, float(np.percentile(ped[ped > 0], 90))) if (ped > 0).any()
                                 else np.zeros_like(self.road))

    # ------------------------------------------------------------------ helpers
    def _poly(self, pts) -> np.ndarray:
        return (np.asarray(pts, np.float64) * [self.w, self.h]).astype(np.float32).reshape(-1, 1, 2)

    def cell(self, p) -> tuple:
        gx = int(np.clip(p[0] / self.w * GX, 0, GX - 1))
        gy = int(np.clip(p[1] / self.h * GY, 0, GY - 1))
        return gy, gx

    def _in_polys(self, key: str, p) -> bool:
        return any(cv2.pointPolygonTest(poly, (float(p[0]), float(p[1])), False) >= 0
                   for poly in self.polys[key])

    @property
    def has_crossings(self) -> bool:
        return bool(self.polys["crossings"])

    def on_road(self, p, core: bool = False) -> bool:
        if self.polys["road"]:
            return self._in_polys("road", p)
        gy, gx = self.cell(p)
        return bool(self.road_core[gy, gx] if core else self.road[gy, gx])

    def in_crossing(self, p, margin: bool = False) -> bool:
        """Inside a pedestrian crossing; `margin` also accepts the kerb-side band around it."""
        if self.has_crossings:
            if not margin:
                return self._in_polys("crossings", p)
            tol = float(self.geometry.get("crossing_margin", 0.02)) * self.h
            return any(cv2.pointPolygonTest(poly, (float(p[0]), float(p[1])), True) >= -tol
                       for poly in self.polys["crossings"])
        gy, gx = self.cell(p)
        return bool(self.crossing_learned[gy, gx])

    def against_flow(self, p, vel, min_dominance: float, min_support: float = 8.0) -> Optional[bool]:
        """True if `vel` opposes a strongly dominant flow at p, False if it agrees,
        None if the cell has no clear single direction (intersections, few samples)."""
        gy, gx = self.cell(p)
        if self.dir_support[gy, gx] < min_support or self.dominance[gy, gx] < min_dominance:
            return None
        ang = np.arctan2(vel[1], vel[0])
        b = int(((ang + np.pi) / (2 * np.pi) * NBINS)) % NBINS
        share = self.dir_share[gy, gx]
        own = share[b] + share[(b - 1) % NBINS] + share[(b + 1) % NBINS]
        dom = self.dominant[gy, gx]
        opposite = min((b - dom) % NBINS, (dom - b) % NBINS) >= 3
        return bool(opposite and own < 0.05)

    def flow_direction(self, p) -> int:
        gy, gx = self.cell(p)
        return int(self.dominant[gy, gx])

    def stop_lines(self) -> List[dict]:
        out = []
        for sl in self.geometry.get("stop_lines", []):
            p1 = np.array(sl["p1"], float) * [self.w, self.h]
            p2 = np.array(sl["p2"], float) * [self.w, self.h]
            d = np.array(sl.get("direction", [0, 1]), float) * [self.w, self.h]
            normal = np.array([-(p2 - p1)[1], (p2 - p1)[0]])
            normal /= np.linalg.norm(normal) + 1e-9
            if normal @ d < 0:
                normal = -normal                     # positive side = past the line
            out.append({"p1": p1, "p2": p2, "normal": normal,
                        "queue_depth": float(sl.get("queue_depth", 0.1)) * self.h,
                        "past_depth": float(sl.get("past_depth", 0.07)) * self.h})
        return out


def load_geometry() -> dict:
    if SCENE_JSON.exists():
        with open(SCENE_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def build_scene(tracks: Dict[int, Track], width: int, height: int) -> SceneModel:
    own = SceneStats()
    own.add_tracks(tracks, width, height)
    prior = SceneStats.load(PRIOR_PATH)
    stats = own if prior is None else own.merged(prior, 1.0)
    return SceneModel(stats, width, height, load_geometry())
