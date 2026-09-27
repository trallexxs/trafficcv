"""Rule tests on synthetic trajectories (no video or GPU needed).

A two-way road: three eastbound lanes (y=300..354) and three westbound lanes
(y=381..435) in a 1280x720 frame.
Run:  python -m pytest -q tests
"""
import numpy as np
import pytest

from src import scene as scene_mod
from src.config import CAR, PERSON, RuleConfig
from src.events import (Snapshot, finalize, rule_jaywalking, rule_red_light,
                        rule_stopped_vehicle, rule_wrong_way, stationary_episodes)
from src.scene import SceneModel, SceneStats
from src.tracks import build_tracks

W, H, FPS = 1280, 720, 5.0


def _track(tid, cls, t0, t1, x0, y0, vx, vy=0.0, w=60, h=40, stop=None):
    """Rows of a constant-velocity track; `stop=(ts, te)` holds position between ts and te."""
    rows, x, y = [], x0, y0
    for t in np.arange(t0, t1, 1 / FPS):
        moving = not (stop and stop[0] <= t < stop[1])
        if moving:
            x, y = x + vx / FPS, y + vy / FPS
        if not (0 <= x <= W and 0 <= y <= H):
            break
        rows.append([round(t * 25), t, tid, cls, 0.9, x - w / 2, y - h, x + w / 2, y])
    return rows


def _background(duration=120.0):
    """Normal traffic: a car every 3 s in every lane."""
    rows, tid = [], 1000
    for t0 in np.arange(0, duration, 3.0):
        for k, (y_east, y_west) in enumerate(zip((300, 327, 354), (381, 408, 435))):
            rows += _track(tid, CAR, t0 + 0.4 * k, duration, 0, y_east, 150)
            rows += _track(tid + 1, CAR, t0 + 1.5 + 0.4 * k, duration, W, y_west, -150)
            tid += 2
    return rows


def _scene(rows, geometry=None):
    tracks = build_tracks(np.asarray(rows, float), FPS)
    stats = SceneStats()
    stats.add_tracks(tracks, W, H)
    return tracks, SceneModel(stats, W, H, geometry or {})


def test_wrong_way_detected():
    rows = _background() + _track(1, CAR, 40, 60, W, 300, -150)     # westbound in eastbound lane
    tracks, scene = _scene(rows)
    events = rule_wrong_way(tracks, scene, RuleConfig())
    assert len(events) == 1 and events[0].tracks == [1]
    assert 40 <= events[0].start < 42


def test_normal_traffic_has_no_wrong_way():
    tracks, scene = _scene(_background())
    assert rule_wrong_way(tracks, scene, RuleConfig()) == []


def test_stopped_vehicle_overtaken():
    rows = _background() + _track(2, CAR, 10, 80, 100, 300, 150, stop=(14, 50))
    tracks, scene = _scene(rows)
    cfg = RuleConfig()
    events = rule_stopped_vehicle(stationary_episodes(tracks, cfg), Snapshot(tracks, 120), scene, cfg, 120)
    assert len(events) == 1
    assert abs(events[0].start - 14) < 1.5 and abs(events[0].end - 50) < 1.5


def test_short_stop_ignored():
    rows = _background() + _track(2, CAR, 10, 80, 100, 300, 150, stop=(14, 20))
    tracks, scene = _scene(rows)
    cfg = RuleConfig()
    assert rule_stopped_vehicle(stationary_episodes(tracks, cfg), Snapshot(tracks, 120), scene, cfg, 120) == []


def test_jaywalking_outside_crossing_only():
    crossing = {"crossings": [[[0.70, 0.30], [0.78, 0.30], [0.78, 0.70], [0.70, 0.70]]]}
    rows = _background()
    rows += _track(3, PERSON, 30, 45, 400, 250, 0, 40, w=20, h=50)      # crosses mid-block
    rows += _track(4, PERSON, 60, 75, 950, 250, 0, 40, w=20, h=50)      # uses the crossing
    tracks, scene = _scene(rows, crossing)
    events = rule_jaywalking(tracks, scene, RuleConfig())
    assert [e.tracks for e in events] == [[3]]


def test_red_light_runner():
    # stop line across the eastbound lane at x=640; two cars wait at it, one drives through
    geometry = {"stop_lines": [{"p1": [0.5, 0.35], "p2": [0.5, 0.56], "direction": [1, 0],
                                "queue_depth": 0.15, "past_depth": 0.05}]}
    rows = _background(40)
    rows += _track(5, CAR, 0, 60, 560, 330, 0)                          # waiting
    rows += _track(6, CAR, 0, 60, 590, 380, 0)                          # waiting
    rows += _track(7, CAR, 20, 40, 300, 360, 150)                       # runs the red
    tracks, scene = _scene(rows, geometry)
    red, _ = rule_red_light(tracks, scene, RuleConfig())
    assert 7 in [tid for e in red for tid in e.tracks]


def test_finalize_merges_and_clips():
    from src.events import Event
    cfg = RuleConfig()
    ev = [Event(1, 5, "jaywalking"), Event(5.5, 9, "jaywalking"), Event(20, 20.3, "jaywalking"),
          Event(-1, 3, "wrong_way"), Event(10, 12, "fire_smoke")]
    out = finalize(ev, 30.0, cfg)
    assert [e.as_list() for e in out] == [[0.0, 3.0, "wrong_way"], [1.0, 9.0, "jaywalking"]]


@pytest.fixture(autouse=True)
def _no_prior(monkeypatch, tmp_path):
    monkeypatch.setattr(scene_mod, "PRIOR_PATH", tmp_path / "none.npz")
