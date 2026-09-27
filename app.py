"""Team website + live demo (Gradio). Runs locally or as a Hugging Face Space.

    python app.py            # http://127.0.0.1:7860
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

os.environ.setdefault("TRAFFIC_DEMO", "1")      # CPU-friendly settings for the demo

import gradio as gr  # noqa: E402
import pandas as pd  # noqa: E402

from src.offline import process_video, render, summary  # noqa: E402
from src.viz import bar_figure, counts_figure, timeline_figure  # noqa: E402

ROOT = Path(__file__).resolve().parent
ASSETS = Path(os.environ.get("SITE_ASSETS", ROOT / "site_assets"))   # written by scripts/process_samples.py
EDA = ASSETS / "eda"
REPO_URL = os.environ.get("REPO_URL", "https://github.com/trallexxs/trafficcv")
MAX_SECONDS = 120
MAX_MB = 200

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
.gradio-container {max-width: 1180px !important; margin: auto}
.pipeline {display:flex; flex-wrap:wrap; gap:8px; align-items:stretch; margin:8px 0 16px}
.pipeline .box {flex:1 1 150px; border:1px solid var(--border-color-primary); border-radius:10px;
  padding:10px 12px; background:var(--block-background-fill)}
.pipeline .box b {display:block; margin-bottom:4px}
.pipeline .learned {border-left:4px solid #0090ff} .pipeline .rules {border-left:4px solid #30a46c}
.pipeline .arrow {align-self:center; font-size:20px; color:var(--body-text-color-subdued)}
.team {display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:12px}
.team .card {border:1px solid var(--border-color-primary); border-radius:12px; padding:14px}
.kpi {display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px}
.kpi div {border:1px solid var(--border-color-primary); border-radius:10px; padding:10px}
.kpi b {font-size:22px; display:block}
"""


# ------------------------------------------------------------------- demo
def _trim(src: str) -> str:
    """Keep the first MAX_SECONDS of the upload (stream copy, no re-encode)."""
    import imageio_ffmpeg
    dst = tempfile.mktemp(suffix=".mp4")
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-y", "-i", src, "-t", str(MAX_SECONDS),
           "-c", "copy", "-an", dst]
    ok = subprocess.run(cmd, capture_output=True).returncode == 0 and Path(dst).exists()
    return dst if ok else src


def run_demo(video_path, progress=gr.Progress()):
    if not video_path:
        raise gr.Error("Upload an .mp4 first.")
    if Path(video_path).stat().st_size > MAX_MB * 1024 * 1024:
        raise gr.Error(f"File is larger than {MAX_MB} MB.")
    progress(0.02, desc="Preparing video")
    path = _trim(video_path)
    work = Path(tempfile.mkdtemp())
    p = process_video(path, work, shared_tracker=True,
                      progress=lambda f, msg: progress(0.05 + 0.8 * f, desc=msg))
    progress(0.88, desc="Rendering annotated video")
    out_video = render(p, work / "annotated.mp4")
    s = summary(p)
    df = pd.DataFrame([{"start (s)": e[0], "end (s)": e[1], "event": e[2], "track ids": ", ".join(map(str, e[3]))}
                       for e in s["events"]]) if s["events"] else pd.DataFrame(
        {"result": ["No events detected in this clip."]})
    js = work / "events.json"
    js.write_text(json.dumps({"video": Path(video_path).name, "events": [e[:3] for e in s["events"]],
                              "risk": s["risk"]}, indent=1))
    counts = ", ".join(f"{v} {k}" for k, v in s["n_tracks"].items() if v)
    info = (f"**{Path(video_path).name}** · {s['resolution'][0]}×{s['resolution'][1]} · {s['fps']:.1f} fps · "
            f"{s['duration']:.1f} s analysed · tracked: {counts or 'nothing'} · "
            f"{len(s['events'])} event(s)")
    return str(out_video), timeline_figure(s["events"], s["duration"], s["risk"]), df, str(js), info


# ---------------------------------------------------------------- samples
def sample_names():
    return sorted(p.stem for p in ASSETS.glob("*.json")) if ASSETS.exists() else []


def show_sample(name):
    if not name:
        return None, None, None
    s = json.loads((ASSETS / f"{name}.json").read_text())
    video = ASSETS / f"{name}.mp4"
    df = pd.DataFrame([{"start (s)": e[0], "end (s)": e[1], "event": e[2], "note": e[4]} for e in s["events"]]) \
        if s["events"] else pd.DataFrame({"result": ["No events detected."]})
    return (str(video) if video.exists() else None,
            timeline_figure(s["events"], s["duration"], s["risk"], title=name), df)


# -------------------------------------------------------------------- page
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


def eda_section():
    gr.Markdown("### Scene layout\nOne still frame of the camera with the geometry we traced: two zebra "
                "crossings, the main-avenue stop line and the queue zone used to infer the red phase.")
    gr.Image(str(ROOT / "assets" / "scene_layout.jpg"), show_label=False, interactive=False)
    report_path = EDA / "eda.json"
    if not report_path.exists():
        gr.Markdown("*Statistics of the sample videos appear here after `scripts/process_samples.py` is run.*")
        return
    rep = json.loads(report_path.read_text())
    vids = rep["videos"]
    gr.Markdown("### The sample videos")
    gr.Dataframe(pd.DataFrame([{k: v[k] for k in ("video", "resolution", "fps", "duration_s", "frames",
                                                    "brightness_mean")} | v["tracks"] for v in vids]),
                 interactive=False)
    total_min = sum(v["duration_s"] for v in vids) / 60
    n_veh = sum(v["tracks"].get(k, 0) for v in vids for k in ("car", "bus", "truck", "motorcycle"))
    n_ped = sum(v["tracks"].get("person", 0) for v in vids)
    gr.HTML(f'<div class="kpi"><div><b>{len(vids)}</b>videos</div><div><b>{total_min:.1f} min</b>footage</div>'
            f'<div><b>{n_veh}</b>vehicle tracks</div><div><b>{n_ped}</b>pedestrian tracks</div></div>')
    with gr.Row():
        gr.Image(str(EDA / "heat_vehicles.jpg"), label="Where vehicles are (log density)")
        gr.Image(str(EDA / "heat_pedestrians.jpg"), label="Where pedestrians walk")
    with gr.Row():
        gr.Image(str(EDA / "direction_field.jpg"), label="Learned traffic direction (green = one dominant heading)")
        gr.Image(str(EDA / "trajectories.jpg"), label="Trajectories, coloured by heading (orange = pedestrians)")
    gr.Image(str(EDA / "heat_stationary.jpg"), label="Where vehicles stand still (queues, parking)")
    for v in vids:
        gr.Plot(counts_figure(v["counts"], f"{v['video']}: new objects over time"))
    gr.Plot(bar_figure(rep["speed_bins"][:-1], rep["vehicle_speed_hist"], "Moving-vehicle speed distribution",
                       "body-lengths / s", "samples"))
    gr.Markdown("**Findings that shaped the solution:** strong perspective (vehicle size varies ~5× from "
                "the bottom to the top of the frame), so all thresholds are relative to object size; a long "
                "standing queue at the stop line every signal cycle, so standing still alone is not an "
                "event and queues must be excluded from stopped_vehicle; pedestrians concentrate on the two "
                "crossings, so any pedestrian elsewhere on the road core is informative.")


with gr.Blocks(title="Traffic Event Detection · WIUT Hackathon") as demo:
    gr.Markdown("# 🚦 Traffic Event Detection & Accident Anticipation\n"
                "Fixed-camera CCTV → every traffic event as a time segment, plus a live accident-risk score. "
                f"[Code repository]({REPO_URL}) · [predictions_samples.json]({REPO_URL}/blob/main/predictions_samples.json) "
                f"· [weights]({REPO_URL}/tree/main/weights)")
    with gr.Tabs():
        with gr.Tab("Live demo"):
            gr.Markdown(f"Upload an **.mp4** (up to **{MAX_SECONDS // 60} minutes / {MAX_MB} MB**; longer clips are "
                        "cut to the first 2 minutes). The model runs on CPU here (YOLO11-n, 4 fps), so a 2-minute "
                        "clip takes roughly 1–3 minutes. Best results on footage from the competition camera.")
            with gr.Row():
                with gr.Column(scale=1):
                    inp = gr.Video(label="Your video", sources=["upload"])
                    btn = gr.Button("Detect events", variant="primary")
                with gr.Column(scale=1):
                    out_video = gr.Video(label="Annotated playback", interactive=False)
            info = gr.Markdown()
            plot = gr.Plot(label="Event timeline and risk curve")
            table = gr.Dataframe(label="Detected events", interactive=False)
            file = gr.File(label="Download events + risk (JSON)")
            btn.click(run_demo, inp, [out_video, plot, table, file, info])
        with gr.Tab("Sample results"):
            names = sample_names()
            if names:
                gr.Markdown("Annotated versions of every sample video, rendered with our own tooling: boxes and "
                            "track ids, active events (red outline = involved road users) and the Part B risk bar. "
                            "Hover the timeline for exact times.")
                dd = gr.Dropdown(names, value=names[0], label="Sample video")
                sv, sp, st = gr.Video(interactive=False), gr.Plot(), gr.Dataframe(interactive=False)
                dd.change(show_sample, dd, [sv, sp, st])
                demo.load(show_sample, dd, [sv, sp, st])
            else:
                gr.Markdown("Sample-video results will appear here once `scripts/process_samples.py` has been "
                            "run on the organisers' sample videos.")
        with gr.Tab("EDA"):
            eda_section()
        with gr.Tab("Approach"):
            gr.HTML(pipeline_html())
            gr.Markdown(APPROACH_MD)
        with gr.Tab("Report"):
            gr.Markdown(REPORT_MD)
        with gr.Tab("Team"):
            gr.HTML(team_html())

if __name__ == "__main__":
    demo.queue(max_size=8).launch(css=CSS, theme=gr.themes.Soft())
