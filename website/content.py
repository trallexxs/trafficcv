"""Static text and HTML of the team website."""

TEAM = [
    {"name": "Bahriddin Bahodurov", "role": "Team lead · product, website, EDA & report",
     "did": "Problem analysis, scene annotation (crossings, stop line), website content, report.",
     "links": {}},
    {"name": "Amirxon Shomurodov", "role": "CTO · models & pipeline",
     "did": "Detection + tracking pipeline, rules engine, submission packaging.",
     "links": {"GitHub": "https://github.com/amirxon3513-ship-it"}},
    {"name": "Bobur G'opurjonov", "role": "Project manager · evaluation & QA",
     "did": "Timeline, testing on sample videos, failure-case review, deployment.",
     "links": {}},
]

CSS = """
<style>
.pipeline {display:flex; flex-wrap:wrap; gap:8px; align-items:stretch; margin:8px 0 16px}
.pipeline .box {flex:1 1 150px; border:1px solid rgba(128,128,128,.35); border-radius:10px; padding:10px 12px}
.pipeline .box b {display:block; margin-bottom:4px}
.pipeline .learned {border-left:4px solid #0090ff} .pipeline .rules {border-left:4px solid #30a46c}
.pipeline .arrow {align-self:center; font-size:20px; opacity:.6}
.team {display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:12px}
.team .card {border:1px solid rgba(128,128,128,.35); border-radius:12px; padding:14px}
.kpi {display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; margin:8px 0}
.kpi div {border:1px solid rgba(128,128,128,.35); border-radius:10px; padding:10px}
.kpi b {font-size:22px; display:block}
</style>
"""


def pipeline_html() -> str:
    steps = [
        ("learned", "1 · Sampled decoding", "ffmpeg keeps 5 of 25 frames/s, resized to 1280 px"),
        ("learned", "2 · YOLO11-s detector", "COCO-pretrained: person, bicycle, car, motorcycle, bus, truck"),
        ("learned", "3 · ByteTrack", "online multi-object tracking → trajectories with ids"),
        ("rules", "4 · Scene model", "road mask + per-cell traffic direction learned from trajectories; "
                                   "crossings & stop line from the still frame"),
        ("rules", "5 · Event rules", "stopped vehicle, wrong way, jaywalking, accident, congestion, "
                                   "red light, stop line, failure to yield"),
        ("rules", "6 · Post-processing", "merge fragments, drop blips, no same-class overlap"),
    ]
    boxes = '<span class="arrow">→</span>'.join(
        f'<div class="box {k}"><b>{t}</b>{d}</div>' for k, t, d in steps)
    return (f'<div class="pipeline">{boxes}</div>'
            '<p><span style="color:#0090ff">■</span> learned (pretrained, open weights) &nbsp; '
            '<span style="color:#30a46c">■</span> rule-based / geometric</p>')


APPROACH_MD = """
### Problem
A fixed CCTV camera looks at a signalised intersection in Tashkent. For every clip we must return
`[start_sec, end_sec, label]` segments for 14 traffic-event classes (Part A) and, frame by frame,
the probability that an accident starts within 5 s using past frames only (Part B).

### Why detector + tracker + rules
There are **no labels** and the test clips contain events that do not appear in the samples, so a
trained event classifier would have nothing to learn from. Almost every class is defined by
*where a road user is and how it moves* (a vehicle standing still ≥ 10 s, moving against the flow,
a pedestrian on the carriageway outside a crossing), so we extract trajectories with open-weight
models and apply transparent rules on top of them.

### Models and data
| Component | Model | Weights / data | Licence |
|---|---|---|---|
| Detector | Ultralytics YOLO11-s (1280 px) | COCO-pretrained, no fine-tuning | AGPL-3.0 |
| Demo detector | YOLO11-n (800 px, CPU) | COCO-pretrained | AGPL-3.0 |
| Tracker | ByteTrack (Ultralytics implementation) | — | MIT / AGPL-3.0 |

No external training data is used. The scene prior (`weights/scene_prior.npz`) is computed from the
organisers' sample videos by `scripts/process_samples.py`.

### Scene model (what makes it camera-specific)
* **Road mask** – grid cells vehicles actually drive through (moving-vehicle foot points).
* **Direction field** – 8-bin heading histogram per cell. A cell with one dominant heading defines
  the legal direction; intersections (mixed headings) are ignored by the wrong-way rule.
* **Hand-traced geometry** (`configs/scene.json`) – two zebra crossings, the stop line of the
  main avenue and the queue zone behind it. The signal heads face away from the camera, so the
  **red phase is inferred from behaviour**: ≥ 2 vehicles standing in the queue zone for ≥ 4 s.

### Event rules (units are body-lengths so they hold across the perspective)
| Class | Rule |
|---|---|
| stopped_vehicle | stationary ≥ 10 s on the road core, overtaken by ≥ 2 vehicles of the same flow, not in a queue |
| wrong_way | ≥ 2 s and ≥ 2.5 body-lengths moving opposite to a cell's dominant direction |
| jaywalking | pedestrian (not a rider) ≥ 1.5 s well inside the road, away from kerbs/islands and outside the crossings |
| accident | two vehicles in contact, one brakes from ≥ 0.8 to < 35 % of its speed within ~1 s and stays stopped ≥ 3 s |
| congestion | ≥ 5 vehicles of one direction, ≥ 75 % crawling, for ≥ 90 s (longer than a red phase) |
| red_light | vehicle crosses the stop line while the queue beside it keeps waiting (red phase) |
| stop_line | vehicle stops beyond the stop line during the red phase; ends when the queue moves |
| failure_to_yield | vehicle drives through a crossing while a pedestrian is on it within 5 body-lengths |

Classes we cannot detect reliably (near miss, U-turn, illegal turn, solid line, obstacle, fire)
are **not emitted**: a predicted class that never occurs adds a zero to the macro-F1.

### Part B – accident anticipation
An independent, strictly causal tracker (YOLO11-s at 960 px, every 0.2 s) feeds three signals:
closest-point-of-approach between road users (a 2-D time-to-collision), hard braking, and motion
against the learned traffic direction. They are combined as `1 − Π(1 − rᵢ)` with fast-attack /
slow-decay smoothing. Part B never opens the file and never reads Part A output.

### Engineering
* 5 fps analysis, frames downsized inside ffmpeg; Part A stops at 1.4 × video duration (the budget is 3 × for A + B).
* Deterministic: fixed seeds, deterministic cuDNN, online tracker.
* Synthetic-trajectory unit tests for every rule (`tests/`).
"""

REPORT_MD = """
### What we built
A detector + tracker + rules system (YOLO11 + ByteTrack) with a camera-specific scene model, a causal
time-to-collision risk estimator, a one-command submission and this website with a live demo.

### What worked
* **Scale-free thresholds.** Speeds in body-lengths per second make one set of thresholds work for
  near and far vehicles despite the strong perspective.
* **Learning the traffic direction from the video itself.** The wrong-way rule needs no manual lane map.
* **Inferring the red phase from the queue.** The signal heads are not visible, but a standing queue at
  the stop line is.
* **Re-joining occluded stops.** A stopped car hidden by a passing bus gets a new tracker id; merging
  stationary episodes by box overlap keeps it as one event.

### What did not work / known failure cases
* **No labels.** Thresholds are set from the traffic rules and a few clips, not tuned against ground
  truth, so temporal boundaries (important at IoU 0.7) are approximate.
* **Accidents and near misses.** From trajectories alone, a hard brake behind a queue looks like a
  rear-end collision. We keep the accident rule strict and do not emit near_miss.
* **Occlusion in the queue.** Vehicles hidden behind buses lose their ids, and a vehicle that stands
  still in the far lanes can be missed.
* **Right turns during the red phase** could be flagged as red-light running if they cross the traced stop line.
* **2-D time-to-collision** over-estimates risk for vehicles in adjacent lanes seen at a flat angle.

### What we would do next
1. Annotate the sample videos (CVAT) and tune every threshold against `evaluate.py` at IoU 0.3/0.5/0.7.
2. Fine-tune the detector on UA-DETRAC for small, far vehicles.
3. Add a homography to metric ground coordinates for true speeds and time-to-collision.
4. Train a light accident / near-miss classifier on DoTA or CCD clips over our trajectory features.
"""


def team_html() -> str:
    cards = []
    for m in TEAM:
        links = " · ".join(f'<a href="{u}" target="_blank">{k}</a>' for k, u in m["links"].items())
        cards.append(f'<div class="card"><h3 style="margin:0">{m["name"]}</h3><i>{m["role"]}</i>'
                     f'<p>{m["did"]}</p><p>{links}</p></div>')
    return f'<div class="team">{"".join(cards)}</div>'
