"""Team website + live demo (Streamlit). Deployed on Streamlit Community Cloud.

    streamlit run website/streamlit_app.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("TRAFFIC_DEMO", "1")          # CPU-friendly settings for the demo
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from content import APPROACH_MD, CSS, REPORT_MD, pipeline_html, team_html  # noqa: E402
from src.viz import bar_figure, counts_figure, timeline_figure  # noqa: E402

ASSETS = Path(os.environ.get("SITE_ASSETS", ROOT / "site_assets"))  # written by scripts/process_samples.py
EDA = ASSETS / "eda"
REPO_URL = "https://github.com/trallexxs/trafficcv"
MAX_SECONDS = 120
MAX_MB = 200

st.set_page_config(page_title="Traffic Event Detection · WIUT Hackathon", page_icon="🚦", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------- demo
def trim(src: str) -> str:
    """Keep the first MAX_SECONDS of the upload (stream copy, no re-encode)."""
    import imageio_ffmpeg
    dst = tempfile.mktemp(suffix=".mp4")
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-y", "-i", src, "-t", str(MAX_SECONDS),
           "-c", "copy", "-an", dst]
    ok = subprocess.run(cmd, capture_output=True).returncode == 0 and Path(dst).exists()
    return dst if ok else src


def events_table(events, with_ids: bool = True) -> pd.DataFrame:
    if not events:
        return pd.DataFrame({"result": ["No events detected."]})
    rows = []
    for e in events:
        row = {"start (s)": e[0], "end (s)": e[1], "event": e[2]}
        if with_ids:
            row["road users (track ids)"] = ", ".join(map(str, e[3]))
        rows.append(row)
    return pd.DataFrame(rows)


def run_demo(upload) -> None:
    from src.offline import process_video, render, summary

    work = Path(tempfile.mkdtemp())
    src = work / "upload.mp4"
    src.write_bytes(upload.getbuffer())
    bar = st.progress(0.0, text="Preparing video…")
    path = trim(str(src))
    p = process_video(path, work, shared_tracker=True,
                      progress=lambda f, msg: bar.progress(min(0.05 + 0.8 * f, 0.85), text=msg))
    bar.progress(0.9, text="Rendering annotated video…")
    out = render(p, work / "annotated.mp4")
    bar.progress(1.0, text="Done")
    st.session_state["demo"] = {"video": out.read_bytes(), "summary": summary(p), "name": upload.name}


def demo_tab() -> None:
    st.markdown(f"Upload an **.mp4** (up to **{MAX_SECONDS // 60} minutes / {MAX_MB} MB**; longer clips are cut "
                "to the first 2 minutes). The model runs on a small CPU server (YOLO11-n, 4 fps), so a 2-minute "
                "clip takes a few minutes. Best results on footage from the competition camera.")
    upload = st.file_uploader("Your video", type=["mp4", "mov"], accept_multiple_files=False)
    if upload is not None and upload.size > MAX_MB * 1024 * 1024:
        st.error(f"File is larger than {MAX_MB} MB.")
        return
    if st.button("Detect events", type="primary", disabled=upload is None):
        try:
            run_demo(upload)
        except Exception as exc:              # keep the page alive on unreadable uploads
            st.error(f"Could not process this video: {exc}")
    res = st.session_state.get("demo")
    if not res:
        return
    s = res["summary"]
    counts = ", ".join(f"{v} {k}" for k, v in s["n_tracks"].items() if v)
    st.markdown(f"**{res['name']}** · {s['resolution'][0]}×{s['resolution'][1]} · {s['fps']:.1f} fps · "
                f"{s['duration']:.1f} s · tracked: {counts or 'nothing'} · **{len(s['events'])} event(s)**")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.video(res["video"])
    with c2:
        st.dataframe(events_table(s["events"]), width="stretch", hide_index=True)
        st.download_button("Download events + risk (JSON)", json.dumps(
            {"video": res["name"], "events": [e[:3] for e in s["events"]], "risk": s["risk"]}, indent=1),
            file_name="events.json", mime="application/json")
    st.plotly_chart(timeline_figure(s["events"], s["duration"], s["risk"]), width="stretch")


# ---------------------------------------------------------------- samples
def samples_tab() -> None:
    names = sorted(p.stem for p in ASSETS.glob("*.json")) if ASSETS.exists() else []
    if not names:
        st.info("Annotated sample videos appear here once `scripts/process_samples.py` has been run on the "
                "organisers' sample videos.")
        return
    st.markdown("Annotated versions of every sample video, rendered with our own tooling: boxes and track ids, "
                "active events (red outline = involved road users) and the Part B risk bar.")
    name = st.selectbox("Sample video", names)
    s = json.loads((ASSETS / f"{name}.json").read_text())
    video = ASSETS / f"{name}.mp4"
    c1, c2 = st.columns([1, 1])
    with c1:
        if video.exists():
            st.video(str(video))
    with c2:
        st.dataframe(pd.DataFrame([{"start (s)": e[0], "end (s)": e[1], "event": e[2], "note": e[4]}
                                   for e in s["events"]]) if s["events"] else events_table([]),
                     width="stretch", hide_index=True)
    st.plotly_chart(timeline_figure(s["events"], s["duration"], s["risk"], title=name), width="stretch")


# -------------------------------------------------------------------- EDA
def eda_tab() -> None:
    st.markdown("#### Scene layout\nA still frame of the camera with the geometry we traced: two zebra crossings, "
                "the main-avenue stop line and the queue zone used to infer the red phase (the signal heads face "
                "away from the camera).")
    st.image(str(ROOT / "assets" / "scene_layout.jpg"), width="stretch")
    st.markdown("**Findings that shaped the solution**\n"
                "* Strong perspective: vehicles near the camera look several times larger than far ones, so every "
                "threshold is measured in *body-lengths* instead of pixels.\n"
                "* A long queue stands at the stop line in every signal cycle, so standing still alone is not an "
                "event: queues are excluded from `stopped_vehicle`, and congestion must outlast a red phase.\n"
                "* Pedestrians cross on two marked zebra crossings; anyone walking elsewhere on the carriageway is "
                "a jaywalking candidate.\n"
                "* The signal lamps are not visible, so the red phase is inferred from the waiting queue.")
    report_path = EDA / "eda.json"
    if not report_path.exists():
        st.info("Statistics of the sample videos (heatmaps, trajectories, counts over time) appear here after "
                "`scripts/process_samples.py` is run.")
        return
    rep = json.loads(report_path.read_text())
    vids = rep["videos"]
    st.markdown("#### The sample videos")
    st.dataframe(pd.DataFrame([{k: v[k] for k in ("video", "resolution", "fps", "duration_s", "frames",
                                                    "brightness_mean")} | v["tracks"] for v in vids]),
                 width="stretch", hide_index=True)
    total_min = sum(v["duration_s"] for v in vids) / 60
    n_veh = sum(v["tracks"].get(k, 0) for v in vids for k in ("car", "bus", "truck", "motorcycle"))
    n_ped = sum(v["tracks"].get("person", 0) for v in vids)
    st.markdown(f'<div class="kpi"><div><b>{len(vids)}</b>videos</div><div><b>{total_min:.1f} min</b>footage</div>'
                f'<div><b>{n_veh}</b>vehicle tracks</div><div><b>{n_ped}</b>pedestrian tracks</div></div>',
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.image(str(EDA / "heat_vehicles.jpg"), caption="Where vehicles are (log density)")
    c2.image(str(EDA / "heat_pedestrians.jpg"), caption="Where pedestrians walk")
    c1.image(str(EDA / "direction_field.jpg"), caption="Learned traffic direction (green = one dominant heading)")
    c2.image(str(EDA / "trajectories.jpg"), caption="Trajectories coloured by heading (orange = pedestrians)")
    st.image(str(EDA / "heat_stationary.jpg"), caption="Where vehicles stand still (queues, parking)")
    for v in vids:
        st.plotly_chart(counts_figure(v["counts"], f"{v['video']}: new objects over time"), width="stretch")
    st.plotly_chart(bar_figure(rep["speed_bins"][:-1], rep["vehicle_speed_hist"], "Moving-vehicle speed distribution",
                               "body-lengths / s", "samples"), width="stretch")


# -------------------------------------------------------------------- page
st.title("🚦 Traffic Event Detection & Accident Anticipation")
st.markdown("Fixed-camera CCTV → every traffic event as a time segment, plus a live accident-risk score. "
            f"[Code repository]({REPO_URL}) · [predictions_samples.json]({REPO_URL}/blob/main/predictions_samples.json)"
            f" · [weights]({REPO_URL}/tree/main/weights)")
tabs = st.tabs(["Live demo", "Sample results", "EDA", "Approach", "Report", "Team"])
with tabs[0]:
    demo_tab()
with tabs[1]:
    samples_tab()
with tabs[2]:
    eda_tab()
with tabs[3]:
    st.markdown(pipeline_html(), unsafe_allow_html=True)
    st.markdown(APPROACH_MD)
with tabs[4]:
    st.markdown(REPORT_MD)
with tabs[5]:
    st.markdown(team_html(), unsafe_allow_html=True)
