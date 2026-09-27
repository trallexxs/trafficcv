"""Part A rules: trajectories + scene model -> event segments.

Every rule returns candidate segments; `finalize` merges fragments, removes
blips and guarantees the harness constraints (no same-class overlap,
0 <= start < end <= duration).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import BICYCLE, MOTORCYCLE, RuleConfig
from .scene import NBINS, SceneModel
from .tracks import Track


@dataclass
class Event:
    start: float
    end: float
    label: str
    tracks: List[int] = field(default_factory=list)
    note: str = ""

    def as_list(self) -> list:
        return [round(float(self.start), 2), round(float(self.end), 2), self.label]


# --------------------------------------------------------------------- helpers
def runs(mask: np.ndarray, t: np.ndarray, merge_gap: float = 0.0) -> List[Tuple[int, int]]:
    """Index ranges [i0, i1] of True runs, merging runs separated by < merge_gap seconds."""
    out: List[Tuple[int, int]] = []
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return out
    s = p = idx[0]
    for i in idx[1:]:
        if i == p + 1 or t[i] - t[p] <= merge_gap:
            p = i
            continue
        out.append((s, p))
        s = p = i
    out.append((s, p))
    return out


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def angle_bin(v: np.ndarray) -> int:
    return int(((np.arctan2(v[1], v[0]) + np.pi) / (2 * np.pi) * NBINS)) % NBINS


def bin_dist(a: int, b: int) -> int:
    return min((a - b) % NBINS, (b - a) % NBINS)


@dataclass
class Episode:
    """A vehicle standing still, possibly spanning several track ids."""
    t0: float
    t1: float
    box: np.ndarray
    foot: np.ndarray
    size: float
    tracks: List[int]


def stationary_episodes(tracks: Dict[int, Track], cfg: RuleConfig) -> List[Episode]:
    eps: List[Episode] = []
    for tr in tracks.values():
        if not tr.is_vehicle:
            continue
        still = tr.speed < cfg.stationary_speed
        for i0, i1 in runs(still, tr.t, merge_gap=0.6):
            if tr.t[i1] - tr.t[i0] < 2.0:
                continue
            eps.append(Episode(tr.t[i0], tr.t[i1], np.median(tr.box[i0:i1 + 1], 0),
                               np.median(tr.foot[i0:i1 + 1], 0), float(np.median(tr.size[i0:i1 + 1])),
                               [tr.tid]))
    # Re-join episodes of the same parked object split by occlusion / id switches.
    eps.sort(key=lambda e: e.t0)
    merged: List[Episode] = []
    for e in eps:
        for m in merged:
            if box_iou(m.box, e.box) > 0.4 and -1.0 < e.t0 - m.t1 < 10.0:
                m.t1 = max(m.t1, e.t1)
                m.tracks += e.tracks
                break
        else:
            merged.append(e)
    return merged


class Snapshot:
    """Positions/speeds of all tracks resampled on a 1 s grid, for crowd-level rules."""

    def __init__(self, tracks: Dict[int, Track], duration: float, step: float = 1.0):
        self.times = np.arange(0.0, max(duration, step), step)
        self.items: List[List[Tuple[Track, int]]] = [[] for _ in self.times]
        for tr in tracks.values():
            k0 = int(np.ceil(tr.t[0] / step))
            k1 = int(np.floor(tr.t[-1] / step))
            for k in range(max(k0, 0), min(k1, len(self.times) - 1) + 1):
                self.items[k].append((tr, tr.index_at(self.times[k])))


# ----------------------------------------------------------------------- rules
def rule_stopped_vehicle(eps: Sequence[Episode], snap: Snapshot, scene: SceneModel,
                         cfg: RuleConfig, duration: float) -> List[Event]:
    out = []
    for e in eps:
        dur = e.t1 - e.t0
        if dur < cfg.stopped_min_s or not scene.on_road(e.foot, core=True):
            continue
        if e.t0 < 1.0 and e.t1 > duration - 1.0:
            continue                                         # parked for the whole clip
        flow = scene.flow_direction(e.foot)
        k0, k1 = int(e.t0), min(int(e.t1), len(snap.times) - 1)
        queued, overtakers = 0, set()
        for k in range(k0, k1 + 1):
            near_still = 0
            for tr, i in snap.items[k]:
                if not tr.is_vehicle or tr.tid in e.tracks:
                    continue
                d = np.linalg.norm(tr.foot[i] - e.foot) / e.size
                if tr.speed[i] < cfg.crawl_speed and d < 2.5:
                    near_still += 1
                elif tr.speed[i] > 1.0 and d < 6.0 and bin_dist(angle_bin(tr.vel[i]), flow) <= 1:
                    overtakers.add(tr.tid)
            queued += near_still > 0
        queue_frac = queued / max(1, k1 - k0 + 1)
        if queue_frac > 0.5:
            continue                                         # waiting in a queue / at a signal
        if len(overtakers) >= 2 or dur >= cfg.stopped_long_s:
            out.append(Event(e.t0, e.t1, "stopped_vehicle", list(e.tracks),
                             f"overtaken by {len(overtakers)}"))
    return out


def rule_congestion(snap: Snapshot, scene: SceneModel, cfg: RuleConfig) -> List[Event]:
    groups = range(NBINS // 2)                               # 4 coarse flow directions
    jam = np.zeros((len(groups), len(snap.times)), bool)
    for k, items in enumerate(snap.items):
        per_group: Dict[int, List[float]] = {}
        for tr, i in items:
            if tr.is_vehicle and scene.on_road(tr.foot[i], core=True):
                per_group.setdefault(scene.flow_direction(tr.foot[i]) // 2, []).append(tr.speed[i])
        for g, speeds in per_group.items():
            speeds = np.asarray(speeds)
            if len(speeds) >= cfg.congestion_min_vehicles and \
                    np.mean(speeds < cfg.crawl_speed) >= cfg.congestion_slow_frac:
                jam[g, k] = True
    out = []
    for g in groups:
        for i0, i1 in runs(jam[g], snap.times, merge_gap=5.0):
            if snap.times[i1] - snap.times[i0] >= cfg.congestion_min_s:
                out.append(Event(snap.times[i0], snap.times[i1] + 1.0, "congestion", note=f"dir {g}"))
    return out


def rule_wrong_way(tracks: Dict[int, Track], scene: SceneModel, cfg: RuleConfig) -> List[Event]:
    out = []
    for tr in tracks.values():
        if not tr.is_vehicle or tr.duration < cfg.wrongway_min_s:
            continue
        verdict = np.full(len(tr.t), -1)                     # -1 unknown, 0 ok, 1 against flow
        for i in range(len(tr.t)):
            if tr.speed[i] < 0.6:
                continue
            v = scene.against_flow(tr.foot[i], tr.vel[i], cfg.wrongway_min_dominance)
            if v is not None:
                verdict[i] = int(v)
        for i0, i1 in runs(verdict == 1, tr.t, merge_gap=1.5):
            seg = verdict[i0:i1 + 1]
            known = seg[seg >= 0]
            disp = np.linalg.norm(tr.foot[i1] - tr.foot[i0]) / np.median(tr.size[i0:i1 + 1])
            if tr.t[i1] - tr.t[i0] >= cfg.wrongway_min_s and disp >= cfg.wrongway_min_disp \
                    and known.mean() >= 0.8:
                out.append(Event(tr.t[i0], tr.t[i1], "wrong_way", [tr.tid]))
    return out


def _rider_mask(tr: Track, snap_by_time) -> np.ndarray:
    """Persons sitting on a bicycle/motorcycle are riders, not pedestrians."""
    mask = np.zeros(len(tr.t), bool)
    for i, t in enumerate(tr.t):
        for other in snap_by_time(t):
            if other.cls in (BICYCLE, MOTORCYCLE):
                j = other.index_at(t)
                if abs(other.t[j] - t) < 0.3 and box_iou(tr.box[i], other.box[j]) > 0.1:
                    mask[i] = True
                    break
    return mask


def rule_jaywalking(tracks: Dict[int, Track], scene: SceneModel, cfg: RuleConfig) -> List[Event]:
    bikes = [tr for tr in tracks.values() if tr.cls in (BICYCLE, MOTORCYCLE)]

    def bikes_at(t):
        return [b for b in bikes if b.t[0] - 0.3 <= t <= b.t[-1] + 0.3]

    out = []
    for tr in tracks.values():
        if not tr.is_person or tr.duration < cfg.jaywalk_min_s:
            continue
        on = np.array([scene.on_road(p, core=True) and not scene.in_crossing(p, margin=True)
                       for p in tr.foot])
        if not on.any():
            continue
        on &= ~_rider_mask(tr, bikes_at)
        for i0, i1 in runs(on, tr.t, merge_gap=1.0):
            if tr.t[i1] - tr.t[i0] >= cfg.jaywalk_min_s:
                out.append(Event(tr.t[i0], tr.t[i1], "jaywalking", [tr.tid]))
    return out


def rule_accident(tracks: Dict[int, Track], cfg: RuleConfig) -> List[Event]:
    """Two vehicles come into contact and the moving one stops abruptly and stays stopped."""
    veh = [tr for tr in tracks.values() if tr.is_vehicle and tr.duration > 2.0]
    out = []
    for a in veh:
        found = False
        for i in range(len(a.t)):
            t = a.t[i]
            if found or a.t[-1] < t + cfg.accident_hold_s:
                break
            b0, b1, a0, a1, h1 = np.searchsorted(
                a.t, [t - 2.0, t - 0.4, t + 0.8, t + 1.2, t + 1.0 + cfg.accident_hold_s])
            if b1 - b0 < 3 or a1 <= a0 or h1 - a0 < 3:
                continue
            v0 = np.median(a.speed[b0:b1])
            if v0 < cfg.accident_min_speed or a.speed[a0:a1].max() > cfg.accident_decel_ratio * v0:
                continue
            if a.speed[a0:h1].max() > 0.3:
                continue
            for b in veh:
                if b.tid == a.tid or not (b.t[0] <= t <= b.t[-1]):
                    continue
                j = b.index_at(t)
                s = 0.5 * (a.size[i] + b.size[j])
                if np.linalg.norm(a.foot[i] - b.foot[j]) < 1.1 * s and box_iou(a.box[i], b.box[j]) > 0.05:
                    stop = t + 1.0
                    jb = b.index_at(stop)
                    if b.speed[jb] > 0.3 and b.t[-1] > stop:  # other party still moving: wait for it
                        later = np.flatnonzero((b.t > stop) & (b.speed < 0.3))
                        stop = b.t[later[0]] if len(later) else min(b.t[-1], t + 6.0)
                    out.append(Event(max(0.0, t - 0.4), stop, "accident", [a.tid, b.tid]))
                    found = True                              # one accident per striking track
                    break
    return out


def red_phase(tracks: Dict[int, Track], sl: dict, snap: Snapshot, cfg: RuleConfig) -> np.ndarray:
    """Per snapshot time: True while the approach is held at red.

    The signal heads face away from the camera, so the phase is read from the
    traffic itself: >= 2 vehicles standing still in the queue zone right behind
    the stop line means the approach has a red signal.
    """
    p1, p2, n = sl["p1"], sl["p2"], sl["normal"]
    seg = p2 - p1
    held = np.zeros(len(snap.times), bool)
    for k, items in enumerate(snap.items):
        waiting = 0
        for tr, i in items:
            if not tr.is_vehicle or tr.speed[i] >= cfg.stationary_speed * 2:
                continue
            q = tr.foot[i] - p1
            along = q @ seg / (seg @ seg)
            if -sl["queue_depth"] <= q @ n < 0 and -0.05 <= along <= 1.05:
                waiting += 1
        held[k] = waiting >= 2
    # a phase shorter than 4 s is noise (a vehicle braking briefly)
    for i0, i1 in runs(held, snap.times):
        if snap.times[i1] - snap.times[i0] < 4.0:
            held[i0:i1 + 1] = False
    return held


def rule_red_light(tracks: Dict[int, Track], scene: SceneModel, cfg: RuleConfig) -> Tuple[List[Event], List[Event]]:
    """red_light: a vehicle crosses the stop line while the queue beside it keeps waiting.
    stop_line: a vehicle stops beyond the stop line during the red phase."""
    red, stopline = [], []
    if not tracks:
        return red, stopline
    snap = Snapshot(tracks, max(tr.t[-1] for tr in tracks.values()) + 1.0, step=0.5)
    for sl in scene.stop_lines():
        held = red_phase(tracks, sl, snap, cfg)

        def is_red(t0: float, t1: float) -> bool:
            k0 = int(np.clip(np.floor(t0 / 0.5), 0, len(held) - 1))
            k1 = int(np.clip(np.ceil(t1 / 0.5), 0, len(held) - 1))
            return bool(held[k0:k1 + 1].all())

        p1, p2, n = sl["p1"], sl["p2"], sl["normal"]
        seg = p2 - p1
        for tr in tracks.values():
            if not tr.is_vehicle:
                continue
            q = tr.foot - p1
            side = q @ n
            along = q @ seg / (seg @ seg)
            within = (along > -0.05) & (along < 1.05)
            for c in np.flatnonzero((side[:-1] < 0) & (side[1:] >= 0) & within[1:])[:1]:
                t = tr.t[c + 1]
                if tr.speed[c + 1] > 0.6 and is_red(t - 1.5, t + 2.5):
                    red.append(Event(t, min(tr.t[-1], t + 10.0), "red_light", [tr.tid]))
            past = (side > 0) & (side < sl["past_depth"]) & within & (tr.speed < cfg.stationary_speed)
            for i0, i1 in runs(past, tr.t, merge_gap=0.6):
                t0 = tr.t[i0]
                if tr.t[i1] - t0 < 2.0 or not is_red(t0, t0 + 1.0):
                    continue
                k = int(t0 / 0.5)
                green = np.flatnonzero(~held[k:])
                t_end = snap.times[k + green[0]] if len(green) else tr.t[i1]
                stopline.append(Event(t0, max(t_end, t0 + 1.0), "stop_line", [tr.tid]))
    return red, stopline


def rule_failure_to_yield(tracks: Dict[int, Track], scene: SceneModel) -> List[Event]:
    if not scene.has_crossings:
        return []
    peds = [tr for tr in tracks.values() if tr.is_person]
    out = []
    for tr in tracks.values():
        if not tr.is_vehicle:
            continue
        inside = np.array([scene.in_crossing(p) for p in tr.foot]) & (tr.speed > 0.5)
        for i0, i1 in runs(inside, tr.t, merge_gap=0.5):
            t0, t1 = tr.t[i0], tr.t[i1]
            for p in peds:
                m = np.flatnonzero((p.t >= t0 - 0.5) & (p.t <= t1))
                if any(scene.in_crossing(p.foot[j]) and
                       np.linalg.norm(p.foot[j] - tr.foot[tr.index_at(p.t[j])]) < 5 * tr.size[i0]
                       for j in m):
                    out.append(Event(t0, t1 + 0.2, "failure_to_yield", [tr.tid, p.tid]))
                    break
    return out


# -------------------------------------------------------------- post-process
def finalize(events: List[Event], duration: float, cfg: RuleConfig,
             enabled: Optional[Sequence[str]] = None) -> List[Event]:
    enabled = set(enabled if enabled is not None else cfg.enabled)
    out: List[Event] = []
    for label in sorted({e.label for e in events} & enabled):
        segs = sorted((e for e in events if e.label == label), key=lambda e: e.start)
        merged: List[Event] = []
        for e in segs:
            e = Event(max(0.0, e.start), min(duration, e.end), e.label, list(e.tracks), e.note)
            if merged and e.start <= merged[-1].end + cfg.merge_gap_s:
                m = merged[-1]
                m.end = max(m.end, e.end)
                m.tracks += e.tracks
            else:
                merged.append(e)
        out += [e for e in merged if e.end - e.start >= cfg.min_event_s]
    return sorted(out, key=lambda e: (e.start, e.label))
