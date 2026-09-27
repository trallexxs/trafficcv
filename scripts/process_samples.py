"""Process the sample videos: scene prior, predictions, EDA and annotated renders.

    python scripts/process_samples.py --videos samples --out outputs

Produces
  weights/scene_prior.npz          direction field / road mask learned from the samples
  predictions_samples.json         same format as run_submission.py output
  site_assets/<video>.json         events + risk curve per video (website)
  site_assets/<video>.mp4          annotated video (website)
  site_assets/eda/                 EDA images + eda.json (website)
  outputs/work/                    proxy videos (intermediate, not committed)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.eda import write_eda  # noqa: E402
from src.offline import expand_risk, process_video, render, summary  # noqa: E402
from src.pipeline import detect  # noqa: E402
from src.scene import PRIOR_PATH, SceneStats  # noqa: E402
from src.tracks import build_tracks  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="samples")
    ap.add_argument("--out", default="outputs", help="folder for intermediate files")
    ap.add_argument("--team", default="all-in-one")
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()

    videos = sorted(p for p in Path(args.videos).iterdir() if p.suffix.lower() in (".mp4", ".mov", ".avi"))
    if not videos:
        sys.exit(f"no videos in {args.videos}")
    out = Path(args.out)
    site = ROOT / "site_assets"
    site.mkdir(parents=True, exist_ok=True)

    processed = []
    for v in videos:
        t0 = time.time()
        print(f"[1/3] analysing {v.name} ...", flush=True)
        p = process_video(str(v), out / "work",
                          progress=lambda f, msg: print(f"    {v.name}: {msg}", flush=True))
        print(f"    done in {time.time() - t0:.0f} s ({p.analysis.meta.duration:.0f} s of video)", flush=True)
        processed.append(p)

    print("[2/3] building scene prior from all samples", flush=True)
    stats = SceneStats()
    for p in processed:
        stats.add_tracks(build_tracks(p.analysis.table, p.analysis.fps_analysed), p.analysis.width, p.analysis.height)
    stats.save(PRIOR_PATH)

    print("[3/3] events, renders, EDA", flush=True)
    preds = {"team": args.team, "videos": {}}
    for p in processed:
        p.detection = detect(p.analysis)            # re-run rules with the prior in place
        name = Path(p.analysis.meta.path).name
        preds["videos"][name] = {"events": [e.as_list() for e in p.detection.events], "risk": expand_risk(p)}
        with open(site / f"{Path(name).stem}.json", "w") as f:
            json.dump(summary(p), f)
        if not args.no_render:
            render(p, site / f"{Path(name).stem}.mp4")
        print(f"    {name}: {len(p.detection.events)} events", flush=True)
    with open(ROOT / "predictions_samples.json", "w") as f:
        json.dump(preds, f)
    write_eda(processed, stats, site / "eda")
    print("finished ->", ROOT / "predictions_samples.json", "and", site)


if __name__ == "__main__":
    main()
