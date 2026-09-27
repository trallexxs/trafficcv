"""Central configuration: model paths, sampling rates and rule thresholds.

All thresholds are expressed in scale-free units so they do not depend on the
video resolution:
  * positions are in analysis-frame pixels (frame resized to ANALYSIS_WIDTH);
  * speeds are divided by the object's own size (sqrt(w*h)), i.e. "body lengths
    per second", which roughly compensates for perspective.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = ROOT / "weights"
CONFIG_DIR = ROOT / "configs"

CLASSES = [
    "accident", "near_miss", "red_light", "wrong_way", "illegal_u_turn",
    "stopped_vehicle", "jaywalking", "failure_to_yield", "illegal_turn",
    "solid_line_crossing", "stop_line", "congestion", "road_obstacle", "fire_smoke",
]

# COCO ids used by the detector.
PERSON, BICYCLE, CAR, MOTORCYCLE, BUS, TRUCK = 0, 1, 2, 3, 5, 7
VEHICLE_CLASSES = (CAR, MOTORCYCLE, BUS, TRUCK)
DETECT_CLASSES = [PERSON, BICYCLE, CAR, MOTORCYCLE, BUS, TRUCK]
COCO_NAMES = {PERSON: "person", BICYCLE: "bicycle", CAR: "car",
              MOTORCYCLE: "motorcycle", BUS: "bus", TRUCK: "truck"}

SEED = 0


@dataclass
class DetectorConfig:
    weights: str = str(WEIGHTS_DIR / "yolo11s.pt")
    imgsz: int = 1280
    conf: float = 0.25
    iou: float = 0.5
    tracker: str = str(CONFIG_DIR / "bytetrack.yaml")


@dataclass
class PipelineConfig:
    analysis_width: int = 1280          # frames are resized to this width
    target_fps: float = 5.0             # Part A sampling rate (frames/s actually analysed)
    time_budget_ratio: float = 1.4      # stop Part A after this x video duration
    detector: DetectorConfig = field(default_factory=DetectorConfig)


@dataclass
class RuleConfig:
    # Which classes are emitted. A predicted class that never occurs in the test
    # set costs a full class in the macro-F1, so only reliable rules are on.
    # Scene-dependent classes are switched on automatically when configs/scene.json
    # provides the geometry they need (crossings, stop line, signal).
    enabled: tuple = ("accident", "stopped_vehicle", "wrong_way", "jaywalking", "congestion")

    stationary_speed: float = 0.12      # body-lengths/s below which a vehicle is "stopped"
    crawl_speed: float = 0.45           # body-lengths/s below which traffic is "crawling"
    stopped_min_s: float = 10.0         # definition from the task
    stopped_long_s: float = 90.0        # stopped this long counts even without overtaking evidence
    congestion_min_vehicles: int = 5
    congestion_slow_frac: float = 0.75
    congestion_min_s: float = 90.0      # longer than one red phase: a signal queue is not congestion
    wrongway_min_s: float = 2.0
    wrongway_min_disp: float = 2.5      # body lengths
    wrongway_min_dominance: float = 0.65
    jaywalk_min_s: float = 1.5
    accident_decel_ratio: float = 0.35  # speed after / before contact
    accident_min_speed: float = 0.8     # body-lengths/s before contact
    accident_hold_s: float = 3.0        # both vehicles stay (nearly) stopped at least this long
    merge_gap_s: float = 2.0            # merge same-class fragments closer than this
    min_event_s: float = 1.0


@dataclass
class RiskConfig:
    stride_s: float = 0.2               # run the causal detector every 0.2 s (0.5 s without a GPU)
    imgsz: int = 960
    horizon_s: float = 3.0              # constant-velocity look-ahead used for CPA
    ema: float = 0.5


def demo_mode() -> bool:
    """CPU demo on the website: smaller model, lower frame rate."""
    return os.environ.get("TRAFFIC_DEMO", "0") == "1"


def has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def pipeline_config() -> PipelineConfig:
    cfg = PipelineConfig()
    if not demo_mode() and not has_cuda():
        # safety net: without a GPU the full model would blow the 3x time budget
        cfg.target_fps = 3.0
        cfg.detector.weights = str(WEIGHTS_DIR / "yolo11n.pt")
        cfg.detector.imgsz = 960
    if demo_mode():
        cfg.analysis_width = 960
        cfg.target_fps = 4.0
        cfg.time_budget_ratio = 10.0
        cfg.detector.weights = str(WEIGHTS_DIR / "yolo11n.pt")
        cfg.detector.imgsz = 800
    return cfg
