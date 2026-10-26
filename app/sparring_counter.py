from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from sparring import (  # noqa: E402
    Detection,
    FighterTracker,
    SparringConfig,
    analyze,
    build_report,
    summarize,
    torso_color,
)

YOUTUBE_ID = re.compile(r"(?:v=|youtu\.be/|shorts/|embed/)([A-Za-z0-9_-]{11})")


def youtube_id(url):
    match = YOUTUBE_ID.search(url)
    if not match:
        raise ValueError(f"Could not find a YouTube video id in {url}")
    return match.group(1)


def download(url, dest_dir):
    dest_dir.mkdir(parents=True, exist_ok=True)
    vid = youtube_id(url)
    existing = [p for p in dest_dir.glob(f"{vid}.*") if p.suffix not in (".part", ".ytdl")]
    if existing:
        return existing[0], vid
    subprocess.run(
        [
            sys.executable,
            "-m",
            "yt_dlp",
            "-f",
            "bv*[height<=720][vcodec^=avc1]/b[height<=720][ext=mp4]/b[height<=720]",
            "-o",
            str(dest_dir / "%(id)s.%(ext)s"),
            url,
        ],
        check=True,
    )
    files = [p for p in dest_dir.glob(f"{vid}.*") if p.suffix not in (".part", ".ytdl")]
    if not files:
        raise RuntimeError("yt-dlp finished but no video file was found")
    return files[0], vid


def run_pose(video_path, model_name, start_s, end_s, imgsz, device, conf, cfg):
    from ultralytics import YOLO

    model = YOLO(model_name)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    first = int(start_s * fps)
    last = total if end_s is None else min(total, int(end_s * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    tracker = FighterTracker(fps, cfg)
    pairs = []
    for n in range(first, last):
        ok, frame = cap.read()
        if not ok:
            break
        result = model.predict(frame, imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
        dets = []
        if result.keypoints is not None and len(result.boxes):
            kps = result.keypoints.xy.cpu().numpy()
            confs = (
                result.keypoints.conf.cpu().numpy()
                if result.keypoints.conf is not None
                else np.ones(kps.shape[:2])
            )
            boxes = result.boxes.xyxy.cpu().numpy()
            for kp, kc, box in zip(kps, confs, boxes):
                dets.append(Detection(kp=kp, conf=kc, box=box, color=torso_color(frame, kp, kc)))
        pairs.append(tracker.update(dets))
        done = n - first + 1
        if done % 300 == 0:
            print(f"{done}/{last - first} frames", flush=True)
    cap.release()
    return pairs, fps, total / fps


def update_manifest(out_dir, slug, report):
    path = out_dir / "index.json"
    manifest = json.loads(path.read_text()) if path.exists() else {"spars": []}
    entry = {
        "slug": slug,
        "title": report.get("title", slug),
        "youtube": report.get("youtube"),
        "video": report.get("video"),
        "start": report["start"],
        "end": report["end"],
        "counts": report["counts"],
    }
    manifest["spars"] = [s for s in manifest["spars"] if s["slug"] != slug] + [entry]
    path.write_text(json.dumps(manifest, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Count punches for both fighters in a sparring clip")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--url", help="YouTube URL (unlisted works)")
    source.add_argument("--video", help="Local video file")
    parser.add_argument("--title", default=None)
    parser.add_argument("--label-a", default="Fighter A", help="Fighter on the left at the start")
    parser.add_argument("--label-b", default="Fighter B", help="Fighter on the right at the start")
    parser.add_argument("--start", type=float, default=0.0, help="Analysis start, seconds")
    parser.add_argument("--end", type=float, default=None, help="Analysis end, seconds")
    parser.add_argument("--out", default=str(ROOT / "web-demo" / "spars"))
    parser.add_argument("--slug", default=None)
    parser.add_argument("--page-video", default=None, help="Path the web page should play instead of YouTube")
    parser.add_argument("--model", default="yolo11s-pose.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument("--det-conf", type=float, default=0.3)
    parser.add_argument("--min-speed", type=float, default=None)
    parser.add_argument("--hit-margin", type=float, default=None)
    parser.add_argument("--jab-max-arc", type=float, default=None)
    args = parser.parse_args()

    cfg = SparringConfig()
    if args.min_speed is not None:
        cfg.min_speed = args.min_speed
    if args.hit_margin is not None:
        cfg.hit_margin = args.hit_margin
    if args.jab_max_arc is not None:
        cfg.jab_max_arc = args.jab_max_arc

    yt = None
    if args.url:
        video_path, yt = download(args.url, ROOT / "data" / "raw_footage")
    else:
        video_path = Path(args.video)

    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", args.slug or yt or video_path.stem).strip("-")
    pairs, fps, duration = run_pose(
        video_path, args.model, args.start, args.end, args.imgsz, args.device, args.det_conf, cfg
    )
    punches, stances = analyze(pairs, fps, cfg, start_s=args.start)
    end_s = args.end if args.end is not None else duration

    meta = {
        "title": args.title or slug,
        "youtube": yt,
        "video": args.page_video,
    }
    report = build_report(punches, stances, [args.label_a, args.label_b], fps, args.start, end_s, meta)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{slug}.json").write_text(json.dumps(report, separators=(",", ":")))
    update_manifest(out_dir, slug, report)

    counts = summarize(punches)
    for fid, label in zip(("A", "B"), (args.label_a, args.label_b)):
        c = counts[fid]
        print(f"{label}: {c['thrown']} thrown, {c['landed']} landed, {c['jab']} jabs, {c['power']} power")
    print(f"Wrote {out_dir / (slug + '.json')}")


if __name__ == "__main__":
    main()
