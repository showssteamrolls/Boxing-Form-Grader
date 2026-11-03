from __future__ import annotations

import argparse
import hashlib
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
    appearance,
    resolve_identity,
    APPEARANCE_DIMS,
    GIRTH_LEVELS,
    stack_tracks,
    clean_track,
    torso_length,
    L_SHOULDER, R_SHOULDER, L_HIP, R_HIP,
    L_ELBOW, R_ELBOW, L_WRIST, R_WRIST, NOSE, L_EYE, R_EYE, L_EAR, R_EAR,
    HEAD_W, HEAD_H,
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


def _encode(src, dst, height, fps, crf, preset="veryfast"):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return dst
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y", "-i", str(src),
            "-vf", f"scale=-2:{height},fps={fps}",
            "-c:v", "libx264", "-crf", str(crf), "-preset", preset,
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(dst),
        ],
        check=True,
    )
    return dst


def _probe(src):
    cap = cv2.VideoCapture(str(src))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return height, fps


def prepare_video(src, slug, out_dir):
    """Two copies: a 720p/30fps one for analysis (kept out of git) and a 1080p/60fps one for the page.

    Never upscales: a 360p or 30fps source keeps its own size and frame rate.
    """
    src_h, src_fps = _probe(src)
    analysis = _encode(src, ROOT / "data" / "analysis" / f"{slug}.mp4", min(720, src_h), 30, 27)
    raw = _encode(src, ROOT / "data" / "analysis" / f"{slug}.display.mp4", min(1080, src_h), 60 if src_fps > 45 else 30, 28)
    return analysis, raw, out_dir / f"{slug}.mp4"


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
            [np.full(APPEARANCE_DIMS, np.nan) if d.color is None else d.color for _, d in rows]
        ).astype(np.float32) if rows else np.zeros((0, APPEARANCE_DIMS), np.float32),
    )


def load_detections(path, meta):
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=False)
    if json.loads(str(data["meta"])) != meta:
        return None
    frames = [[] for _ in range(int(data["n_frames"]))]
    for k, kp, conf, box, color in zip(data["frame"], data["kp"], data["conf"], data["box"], data["color"]):
        col = None if np.isnan(color).all() else color.astype(float)
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
                dets.append(Detection(kp=kp, conf=kc, box=box, color=appearance(frame, kp, kc, box)))
        frames.append(dets)
        if len(frames) % 200 == 0:
            print(f"{len(frames)} frames processed ({(n - first) / native_fps:.0f}s of video)", flush=True)
    cap.release()
    return frames, native_fps / stride, (width, height), total / native_fps


def needs_appearance_refresh(frames):
    for dets in frames:
        for d in dets:
            if d.color is not None:
                return len(d.color) != APPEARANCE_DIMS
    return False


def refresh_appearance(video_path, frames, start_s, stride):
    cap = cv2.VideoCapture(str(video_path))
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_s * native_fps))
    for k, dets in enumerate(frames):
        if k:
            for _ in range(stride - 1):
                cap.grab()
        ok, frame = cap.read()
        if not ok:
            break
        for d in dets:
            d.color = appearance(frame, d.kp, d.conf, d.box)
    cap.release()


def fighter_height(frames):
    """Typical fighter height in the frame: the median over frames of the tallest person found."""
    tallest = [max(float(d.box[3] - d.box[1]) for d in dets) for dets in frames if dets]
    return float(np.median(tallest)) if tallest else 0.0


def is_background(det, ref_h, cfg):
    return float(det.box[3] - det.box[1]) < cfg.background_share * ref_h


def track_frames(frames, fps, cfg):
    """Track the two fighters. People much shorter than a fighter (onlookers, someone on the floor) are ignored,
    so nothing gets tracked when the fighters walk out of frame."""
    ref_h = fighter_height(frames)
    tracker = FighterTracker(fps, cfg)
    pairs = [tracker.update([d for d in dets if not is_background(d, ref_h, cfg)]) for dets in frames]
    pairs, flips = resolve_identity(pairs)
    print(f"Identity pass (glove color and top brightness): {flips} order changes applied")
    return pairs


def blur_display(raw, dst, frames, pairs, fps, start, size, cfg):
    """Write the page video with everyone who is not a fighter blurred out.

    Blur boxes come from the pose detections of people shorter than a fighter, held for 0.17 s either
    side so a flickering detection does not flicker the blur. Fighter boxes are always left sharp.
    """
    ref_h = fighter_height(frames)
    boxes = [[d.box for d in dets if is_background(d, ref_h, cfg)] for dets in frames]
    sig = hashlib.md5(json.dumps([[np.round(b).astype(int).tolist() for b in bs] for bs in boxes]).encode()).hexdigest()
    stamp = ROOT / "data" / "analysis" / f"{dst.stem}.blur"
    if dst.exists() and stamp.exists() and stamp.read_text() == sig and dst.stat().st_mtime >= raw.stat().st_mtime:
        return
    cap = cv2.VideoCapture(str(raw))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dfps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    sx, sy = width / size[0], height / size[1]
    tmp = dst.with_suffix(".tmp.mp4")
    ff = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", f"{dfps}",
         "-i", "-", "-c:v", "libx264", "-crf", "28", "-preset", "veryfast", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", "-an", str(tmp)],
        stdin=subprocess.PIPE,
    )
    hold = max(1, int(round(0.17 * fps)))
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        a = int(round((i / dfps - start) * fps))
        near = [b for k in range(max(0, a - hold), min(len(boxes), a + hold + 1)) for b in boxes[k]]
        if near and 0 <= a < len(boxes):
            out = frame.copy()
            for x0, y0, x1, y1 in near:
                pw, ph = 0.15 * (x1 - x0), 0.08 * (y1 - y0)
                X0, X1 = int(max(0, (x0 - pw) * sx)), int(min(width, (x1 + pw) * sx))
                Y0, Y1 = int(max(0, (y0 - ph) * sy)), int(min(height, (y1 + ph) * sy))
                if X1 - X0 < 4 or Y1 - Y0 < 4:
                    continue
                k = max(3, ((Y1 - Y0) // 6) | 1)
                out[Y0:Y1, X0:X1] = cv2.GaussianBlur(frame[Y0:Y1, X0:X1], (k, k), 0)
            for d in (pairs[a] if a < len(pairs) else []):
                if d is None:
                    continue
                x0, y0, x1, y1 = d.box
                X0, X1 = int(max(0, x0 * sx)), int(min(width, x1 * sx))
                Y0, Y1 = int(max(0, y0 * sy)), int(min(height, y1 * sy))
                out[Y0:Y1, X0:X1] = frame[Y0:Y1, X0:X1]
            frame = out
        ff.stdin.write(frame.tobytes())
        i += 1
    ff.stdin.close()
    ff.wait()
    cap.release()
    tmp.replace(dst)
    stamp.write_text(sig)


def calibrate_girth(video_path, pairs, fps, cfg, seg_model, slug, samples=60, recalibrate=False):
    """Measure each fighter's body width from person silhouettes on a sample of clear frames.

    Pose points sit on the joints, so they understate a real body (more so for a bigger fighter).
    A segmentation model gives the silhouette; width is read across the torso at five heights from
    hip line to shoulder line, normalised by torso length. A low percentile over the sampled frames
    keeps arms and gloves held out to the side from inflating it. Cached next to the detections.
    """
    path = CACHE_DIR / f"{slug}.girth.json"
    if path.exists() and not recalibrate:
        data = json.loads(path.read_text())
        return [d if d is None else dict(d) for d in data]
    from ultralytics import YOLO

    tracks = [clean_track(t, fps, cfg) for t in stack_tracks(pairs)]
    torsos = [torso_length(t) for t in tracks]
    n = len(tracks[0].kp)
    # frames where the fighter faces the camera (shoulders at their widest spread) and the two boxes do not touch
    wanted = {}
    for i in (0, 1):
        spread = np.abs(tracks[i].kp[:, L_SHOULDER, 0] - tracks[i].kp[:, R_SHOULDER, 0])
        ref = np.nanpercentile(spread, 90) if np.isfinite(spread).sum() > 10 else np.nan
        good = []
        for k in range(n):
            kp = tracks[i].kp[k]
            if np.isnan(kp[[L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]]).any() or not spread[k] >= 0.85 * ref:
                continue
            a, b = tracks[0].box[k], tracks[1].box[k]
            if np.isnan(a).any() or np.isnan(b).any() or (a[2] > b[0] and b[2] > a[0]):
                continue
            good.append(k)
        if len(good) < 5:
            return [None, None]
        for j in np.linspace(0, len(good) - 1, min(samples, len(good))):
            wanted.setdefault(good[int(j)], set()).add(i)
    picks = sorted(wanted)
    model = YOLO(seg_model)
    cap = cv2.VideoCapture(str(video_path))
    widths = [[[] for _ in GIRTH_LEVELS] for _ in (0, 1)]
    head_ws = [[], []]
    for k in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, k)
        ok, frame = cap.read()
        if not ok:
            continue
        res = model(frame, classes=[0], imgsz=960, retina_masks=True, verbose=False)[0]
        if res.masks is None:
            continue
        masks = res.masks.data.cpu().numpy() > 0.5
        boxes = res.boxes.xyxy.cpu().numpy()
        for i in sorted(wanted[k]):
            tb = tracks[i].box[k]
            best, best_iou = None, 0.3
            for m, bx in zip(masks, boxes):
                x0, y0 = max(tb[0], bx[0]), max(tb[1], bx[1])
                x1, y1 = min(tb[2], bx[2]), min(tb[3], bx[3])
                inter = max(0, x1 - x0) * max(0, y1 - y0)
                iou = inter / ((tb[2] - tb[0]) * (tb[3] - tb[1]) + (bx[2] - bx[0]) * (bx[3] - bx[1]) - inter + 1e-9)
                if iou > best_iou:
                    best, best_iou = m, iou
            if best is None:
                continue
            kp = tracks[i].kp[k]
            hips = (kp[L_HIP] + kp[R_HIP]) / 2.0
            sh = (kp[L_SHOULDER] + kp[R_SHOULDER]) / 2.0
            # take the arms out of the mask so they do not read as body width
            best = best.copy().astype(np.uint8)
            th = max(3, int(round(0.14 * torsos[i])))
            for s_j, e_j, w_j in ((L_SHOULDER, L_ELBOW, L_WRIST), (R_SHOULDER, R_ELBOW, R_WRIST)):
                if np.isnan(kp[[s_j, e_j, w_j]]).any():
                    continue
                start = kp[s_j] + 0.4 * (kp[e_j] - kp[s_j])
                cv2.line(best, tuple(int(v) for v in start), tuple(int(v) for v in kp[e_j]), 0, th)
                cv2.line(best, tuple(int(v) for v in kp[e_j]), tuple(int(v) for v in kp[w_j]), 0, th)
            best = best > 0
            # head width across the ear line
            ears = kp[[L_EAR, R_EAR]]
            ref_pts = [kp[j] for j in (NOSE, L_EYE, R_EYE) if not np.isnan(kp[j]).any()]
            if ref_pts:
                hy = int(round(float(np.mean([q[1] for q in ref_pts]))))
                hx = int(round(float(np.mean([q[0] for q in ref_pts]))))
                if 0 <= hy < best.shape[0] and 0 <= hx < best.shape[1] and best[hy, hx]:
                    row = best[hy]
                    lo = hx
                    while lo > 0 and row[lo - 1]:
                        lo -= 1
                    hi = hx
                    while hi < len(row) - 1 and row[hi + 1]:
                        hi += 1
                    head_ws[i].append((hi - lo + 1) / torsos[i])
            for li, lv in enumerate(GIRTH_LEVELS):
                c = hips + lv * (sh - hips)
                x, y = int(round(c[0])), int(round(c[1]))
                if not (0 <= y < best.shape[0] and 0 <= x < best.shape[1]):
                    continue
                row = best[y]
                if not row[x]:
                    near = np.flatnonzero(row[max(0, x - 6): x + 7])
                    if len(near) == 0:
                        continue
                    x = max(0, x - 6) + int(near[len(near) // 2])
                lo = x
                while lo > 0 and row[lo - 1]:
                    lo -= 1
                hi = x
                while hi < len(row) - 1 and row[hi + 1]:
                    hi += 1
                widths[i][li].append((hi - lo + 1) / torsos[i])
    cap.release()
    out = []
    for i in (0, 1):
        ws = []
        for li in range(len(GIRTH_LEVELS)):
            vals = widths[i][li]
            ws.append(float(np.percentile(vals, 35)) if len(vals) >= 5 else None)
        if any(w is None for w in ws):
            out.append(None)
            continue
        floor = 0.85 * (ws[0] + ws[-1]) / 2.0   # arm removal can pinch the middle; a torso does not narrow that much
        entry = {"levels": list(GIRTH_LEVELS), "widths": [float(np.clip(max(w, floor), 0.4, 1.6)) for w in ws]}
        if len(head_ws[i]) >= 5:
            hw = float(np.clip(np.percentile(head_ws[i], 50), 0.8 * HEAD_W, 1.5 * HEAD_W))
            entry["head_w"] = hw
            entry["head_h"] = float(HEAD_H * hw / HEAD_W)
        out.append(entry)
    path.write_text(json.dumps([o for o in out]))
    return out


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
    source.add_argument("--video", help="Local video file (re-encoded to a 720p/30fps analysis copy and a 1080p/60fps web copy)")
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
    parser.add_argument("--arm-clear", type=float, default=None)
    parser.add_argument("--jab-max-arc", type=float, default=None)
    parser.add_argument("--min-reach", type=float, default=None)
    parser.add_argument("--color-weight", type=float, default=None)
    parser.add_argument("--seg-model", default=str(ROOT / "models" / "yolo11s-seg.pt"), help="Person segmentation weights, used to measure each fighter's body width")
    parser.add_argument("--no-girth", action="store_true", help="Skip the silhouette measurement and size the targets from pose points only")
    parser.add_argument("--recalibrate", action="store_true", help="Measure body width again instead of using the cached values")
    args = parser.parse_args()

    cfg = SparringConfig()
    for name in ("min_speed", "arm_clear", "jab_max_arc", "min_reach", "color_weight"):
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
        video_path, display_raw, display_path = prepare_video(src, slug_hint, out_dir)
        page_video = f"spars/{display_path.name}"

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
        if needs_appearance_refresh(frames):
            print("Refreshing appearance features from the video")
            refresh_appearance(video_path, frames, args.start, args.stride)
            save_detections(cache_path, frames, fps, size, meta)
    else:
        frames, fps, size, duration = detect_frames(
            video_path, args.model, args.start, args.end, args.stride, args.imgsz, args.device, args.det_conf
        )
        save_detections(cache_path, frames, fps, size, meta)

    pairs = track_frames(frames, fps, cfg)
    if page_video:
        blur_display(display_raw, display_path, frames, pairs, fps, args.start, size, cfg)
    girth = calibrate_girth(video_path, pairs, fps, cfg, args.seg_model, slug, recalibrate=args.recalibrate) if not args.no_girth else None
    if girth:
        print("Body width (torso lengths, hips to shoulders): " + " | ".join(
            "-" if g is None else " ".join(f"{w:.2f}" for w in g["widths"]) + (f" | head {g['head_w']:.2f}" if 'head_w' in g else "") for g in girth))
    analysis = analyze_full(pairs, fps, cfg, start_s=args.start, girth=girth)
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
