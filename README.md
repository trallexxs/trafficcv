# Traffic Event Detection & Accident Anticipation (WIUT Hackathon, CV track)

Team **ALL-IN-ONE**: Bahriddin Bahodurov · Amirxon Shomurodov · Bobur G'opurjonov

A fixed CCTV camera watches a signalised intersection. For each video we return every traffic event
as `[start_sec, end_sec, label]` (Part A) and, frame by frame and causally, the probability that an
accident starts within 5 s (Part B).

**Code:** https://github.com/trallexxs/trafficcv · **Website and live demo:** `<HF_SPACE_URL>`

## Quick start (judges)

```bash
pip install -r requirements.txt
python run_submission.py --videos /data/test --out predictions.json
python evaluate.py --pred predictions.json --validate-only
```

* All weights are already in `weights/` (YOLO11-s 19 MB, YOLO11-n 5.6 MB, scene prior). No download
  or internet access is needed at run time.
* GPU is used when available (fp16). Without a GPU the code switches to YOLO11-n and a lower frame
  rate so it still fits the time budget.
* Tested on Python 3.11 with `torch 2.6.0` and `ultralytics 8.4.163` from a clean virtual environment.

## Repository layout

```
solution.py              interface: detect_events() and RiskEstimator (thin wrapper around src/)
run_submission.py        organisers' harness (unchanged)
evaluate.py              organisers' metric (unchanged)
requirements.txt         submission runtime
requirements-app.txt     website / sample-processing extras
configs/
  scene.json             hand-traced scene geometry: crossings, stop line, queue zone
  bytetrack.yaml         tracker settings
weights/                 yolo11s.pt, yolo11n.pt, scene_prior.npz
src/
  config.py              every threshold and setting in one place
  video.py               probing + fast sampled decoding (bundled ffmpeg, OpenCV fallback)
  detector.py            YOLO11 + ByteTrack wrapper, seeds
  tracks.py              detection table -> tracks with smoothed kinematics
  scene.py               road mask, direction field, crossings, stop lines
  events.py              Part A rules + post-processing
  pipeline.py            Part A: decode -> detect/track -> scene -> rules
  risk.py                Part B: causal time-to-collision risk estimator
  offline.py, eda.py, viz.py   sample processing, rendering, EDA and charts for the website
scripts/
  process_samples.py     scene prior, predictions_samples.json, annotated videos, EDA
space/                   Hugging Face Space entry point (clones this repo and runs app.py)
app.py                   website + live demo (Gradio)
tests/                   rule tests on synthetic trajectories (pytest)
predictions_samples.json our output on the sample videos
```

## Approach

```
video ─► sampled decoding (5 fps, 1280 px) ─► YOLO11-s ─► ByteTrack ─► trajectories
                                                                         │
             configs/scene.json (crossings, stop line) ─► scene model ◄──┤ road mask + direction field
                                                                         ▼
                                                        rules ─► merge/clean ─► events
```

**Learned (pretrained, open weights):** object detection (YOLO11, COCO) and tracking (ByteTrack).
**Rule-based:** everything after the trajectories. There are no labels and the test set contains
events absent from the samples, so interpretable rules on trajectories generalise better than a
classifier trained on nothing.

**Scale-free units.** Positions are foot points (bottom-centre of the box); speeds are divided by
the object's size `sqrt(w·h)` (body-lengths per second), so one threshold works across the strong
perspective of the camera.

**Scene model** (`src/scene.py`)
* *Road mask*: grid cells (48×27) that moving vehicles pass through.
* *Direction field*: an 8-bin heading histogram per cell. Cells with one dominant heading define
  the legal direction, and intersection cells with mixed headings are ignored.
* *Geometry* from `configs/scene.json`, traced on a still frame: two zebra crossings and the stop
  line of the main avenue. The vehicle signals face away from the camera, so the **red phase is
  inferred from the queue**: at least 2 vehicles standing in the zone behind the stop line for at
  least 4 s.
* A prior over the road mask and direction field, learned from all sample videos
  (`weights/scene_prior.npz`), is merged with the statistics of the video being processed.

**Part A rules** (`src/events.py`, thresholds in `src/config.py`)

| Class | Rule |
|---|---|
| stopped_vehicle | stationary ≥ 10 s on the road core, overtaken by ≥ 2 vehicles of the same flow, not queued behind/beside other stopped vehicles; stops split by occlusion are re-joined |
| wrong_way | moving ≥ 2 s and ≥ 2.5 body-lengths against the dominant heading of the cells it drives through |
| jaywalking | pedestrian (riders excluded) ≥ 1.5 s well inside the road (away from kerbs and islands), outside the crossings and their margin |
| accident | contact between two vehicles, one brakes from ≥ 0.8 to < 35 % of its speed within ~1 s and stays stopped ≥ 3 s; ends when all involved stop |
| congestion | ≥ 5 vehicles of one flow direction, ≥ 75 % crawling, for ≥ 90 s (longer than a red phase) |
| red_light | a vehicle crosses the stop line while the queue keeps waiting (red phase before and after) |
| stop_line | a vehicle stops beyond the stop line during the red phase; ends when the queue moves |
| failure_to_yield | a vehicle drives through a crossing while a pedestrian is on it within 5 body-lengths |

`near_miss`, `illegal_u_turn`, `illegal_turn`, `solid_line_crossing`, `road_obstacle` and `fire_smoke`
are **not emitted**. The metric adds every predicted class to the macro average, so a class we
predict that is absent from the test set would cost a whole class.

Post-processing merges same-class fragments less than 2 s apart, removes segments shorter than 1 s,
clips segments to the video and guarantees no same-class overlap.

**Part B** (`src/risk.py`): an independent causal detector and tracker (YOLO11-s at 960 px, every
0.2 s; the harness streams frames and we never open the file or use Part A output). The risk
combines:
1. closest point of approach between road users under constant velocity (a 2-D time-to-collision):
   `exp(-(d_min/0.5)²) · sigmoid((1.5 − t*)/0.35)`;
2. hard braking (speed drops by more than 70 % within about 0.6 s);
3. motion against the direction field of the scene prior.

These are combined as `1 − Π(1 − r_i)` with fast-attack / slow-decay smoothing. Frames between
detector updates return the last score.

## Engineering and time budget

* ffmpeg (bundled via `imageio-ffmpeg`, works offline) drops unselected frames and resizes kept
  ones before Python sees them; OpenCV is the fallback.
* Part A analyses 5 of 25 frames per second and stops itself at 1.4 × the video duration. Part B
  runs its detector 5 times per second. Together they stay well inside the 3 × duration limit on a
  T4.
* Weights and the model are loaded once per video; a fresh tracker is created per video.

## Determinism

`src/detector.py` fixes Python, NumPy and PyTorch seeds (0) and sets deterministic cuDNN.
ByteTrack and all rules are deterministic, so two runs give the same `predictions.json` up to
floating-point noise from fp16 inference.

## Reproducing the sample results and the website

```bash
pip install -r requirements.txt -r requirements-app.txt
python scripts/process_samples.py --videos samples --out outputs   # prior, predictions_samples.json, renders, EDA
python -m pytest -q tests                                          # rule tests
python app.py                                                      # website on http://127.0.0.1:7860
```

## Datasets, models and licences

| Item | Source | Licence |
|---|---|---|
| YOLO11-s / YOLO11-n weights | Ultralytics, pretrained on COCO | AGPL-3.0 |
| COCO (pretraining of the detector, not used by us directly) | cocodataset.org | CC BY 4.0 |
| ByteTrack implementation | Ultralytics | AGPL-3.0 |
| Organisers' sample videos | WIUT Hackathon | competition use |

No other external data was used. Because Ultralytics is AGPL-3.0, this repository is released
under AGPL-3.0.

## Team

| Member | Role | Contribution |
|---|---|---|
| Bahriddin Bahodurov | Team lead (CEO) | problem analysis, scene annotation, website content, report |
| Amirxon Shomurodov | CTO | detection and tracking pipeline, rules, packaging |
| Bobur G'opurjonov | Project manager | planning, testing on samples, deployment |

## Known limitations

No labels were available, so thresholds come from the traffic rules and visual checks rather than
tuning against `evaluate.py`. From trajectories alone, a hard brake behind a queue can look like a
rear-end collision. Vehicles hidden behind buses lose their track ids. The time-to-collision is
computed in the image plane, not in metric ground coordinates.
