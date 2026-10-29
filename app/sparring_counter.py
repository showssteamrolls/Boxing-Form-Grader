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

from spar_features import build_pose, build_series  # noqa: E402
from sparring import (  # noqa: E402
    Detection,
    FighterTracker,
    SparringConfig,
    analyze_full,
    build_report,
    summarize,
    torso_color,
)

YOUTUBE_ID = re.compile(r"(?:v=|youtu\.be/|shorts/|embed/)([A-Za-z0-9_-]{11})")
CACHE_DIR = ROOT / "data" / "pose_cache"


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


def prepare_video(src, dst, height=720, fps=30):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return dst
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y", "-i", str(src),
            "-vf", f"scale=-2:{height},fps={fps}",
            "-c:v", "libx264", "-crf", "27", "-preset", "veryfast",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(dst),
        ],
        check=True,
    )
    return dst


def save_detections(path, frames, fps, size, meta):
    rows = [(k, d) for k, dets in enumerate(frames) for d in dets]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        n_frames=len(frames),
        fps=fps,
        size=np.array(size),
        meta=json.dumps(meta),
        frame=np.array([k for k, _ in rows], dtype=np.int32),
        kp=np.stack([d.kp for _, d in rows]).astype(np.float32) if rows else np.zeros((0, 17, 2), np.float32),
        conf=np.stack([d.conf for _, d in rows]).astype(np.float32) if rows else np.zeros((0, 17), np.float32),
        box=np.stack([d.box for _, d in rows]).astype(np.float32) if rows else np.zeros((0, 4), np.float32),
        color=np.stack(
            [np.full(3, np.nan) if d.color is None else d.color for _, d in rows]
        ).astype(np.float32) if rows else np.zeros((0, 3), np.float32),
    )


def load_detections(path, meta):
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=False)
    if json.loads(str(data["meta"])) != meta:
        return None
    frames = [[] for _ in range(int(data["n_frames"]))]
    for k, kp, conf, box, color in zip(data["frame"], data["kp"], data["conf"], data["box"], data["color"]):
        col = None if np.isnan(color).any() else color.astype(float)
        frames[int(k)].append(Detection(kp=kp.astype(float), conf=conf.astype(float), box=box.astype(float), color=col))
    return frames, float(data["fps"]), tuple(int(v) for v in data["size"])


def detect_frames(video_path, model_name, start_s, end_s, stride, imgsz, device, conf):
    from ultralytics import YOLO

    model = YOLO(model_name)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    first = int(start_s * native_fps)
    last = total if end_s is None else min(total, int(end_s * native_fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    frames = []
    for n in range(first, last):
        if (n - first) % stride:
            if not cap.grab():
                break
            continue
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
        frames.append(dets)
        if len(frames) % 200 == 0:
            print(f"{len(frames)} frames processed ({(n - first) / native_fps:.0f}s of video)", flush=True)
    cap.release()
    return frames, native_fps / stride, (width, height), total / native_fps


def track_frames(frames, fps, cfg):
    tracker = FighterTracker(fps, cfg)
    return [tracker.update(dets) for dets in frames]


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
    parser = argparse.ArgumentParser(description="Count punches and engineer live metrics for both fighters in a spar")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--url", help="YouTube URL (unlisted works)")
    source.add_argument("--video", help="Local video file (re-encoded to a 720p/30fps web copy first)")
    parser.add_argument("--title", default=None)
    parser.add_argument("--label-a", default="Fighter A", help="Fighter on the left at the start")
    parser.add_argument("--label-b", default="Fighter B", help="Fighter on the right at the start")
    parser.add_argument("--start", type=float, default=0.0, help="Analysis start, seconds")
    parser.add_argument("--end", type=float, default=None, help="Analysis end, seconds")
    parser.add_argument("--out", default=str(ROOT / "web-demo" / "spars"))
    parser.add_argument("--slug", default=None)
    parser.add_argument("--model", default="yolo11s-pose.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--stride", type=int, default=1, help="Analyse every Nth frame")
    parser.add_argument("--device", default=None)
    parser.add_argument("--det-conf", type=float, default=0.3)
    parser.add_argument("--redetect", action="store_true", help="Ignore the cached pose detections")
    parser.add_argument("--min-speed", type=float, default=None)
    parser.add_argument("--hit-margin", type=float, default=None)
    parser.add_argument("--jab-max-arc", type=float, default=None)
    args = parser.parse_args()

    cfg = SparringConfig()
    for name in ("min_speed", "hit_margin", "jab_max_arc"):
        value = getattr(args, name)
        if value is not None:
            setattr(cfg, name, value)

    out_dir = Path(args.out)
    yt = None
    page_video = None
    if args.url:
        video_path, yt = download(args.url, ROOT / "data" / "raw_footage")
    else:
        src = Path(args.video)
        slug_hint = re.sub(r"[^A-Za-z0-9_-]+", "-", args.slug or src.stem).strip("-")
        video_path = prepare_video(src, out_dir / f"{slug_hint}.mp4")
        page_video = f"spars/{video_path.name}"

    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", args.slug or yt or video_path.stem).strip("-")
    meta = {
        "model": args.model, "imgsz": args.imgsz, "stride": args.stride,
        "start": args.start, "end": args.end, "det_conf": args.det_conf,
        "video_size": video_path.stat().st_size,
    }
    cache_path = CACHE_DIR / f"{slug}.npz"
    cached = None if args.redetect else load_detections(cache_path, meta)
    if cached:
        frames, fps, size = cached
        duration = (args.end if args.end is not None else (len(frames) / fps + args.start))
        print(f"Reusing cached detections ({len(frames)} frames)")
    else:
        frames, fps, size, duration = detect_frames(
            video_path, args.model, args.start, args.end, args.stride, args.imgsz, args.device, args.det_conf
        )
        save_detections(cache_path, frames, fps, size, meta)

    pairs = track_frames(frames, fps, cfg)
    analysis = analyze_full(pairs, fps, cfg, start_s=args.start)
    punches, stances = analysis["punches"], analysis["stances"]
    end_s = args.start + len(frames) / fps

    info = {"title": args.title or slug, "youtube": yt, "video": page_video, "size": list(size)}
    report = build_report(punches, stances, [args.label_a, args.label_b], fps, args.start, end_s, info)
    report["version"] = 2
    report["series"] = build_series(analysis, punches, fps, args.start)
    pose = build_pose(analysis, fps, args.start, size[0], size[1])

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{slug}.json").write_text(json.dumps(report, separators=(",", ":")))
    if pose:
        (out_dir / f"{slug}.pose.json").write_text(json.dumps(pose, separators=(",", ":")))
    update_manifest(out_dir, slug, report)

    counts = summarize(punches)
    for fid, label in zip(("A", "B"), (args.label_a, args.label_b)):
        c = counts[fid]
        rate = f"{100 * c['landed'] / c['thrown']:.0f}%" if c["thrown"] else "--"
        print(f"{label}: {c['thrown']} thrown, {c['landed']} landed ({rate}), {c['jab']} jabs, {c['power']} power")
    print(f"Wrote {out_dir / (slug + '.json')}")


if __name__ == "__main__":
    main()
