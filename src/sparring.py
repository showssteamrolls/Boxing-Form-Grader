from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import convolve1d, median_filter
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks

NOSE = 0
L_SHOULDER = 5
R_SHOULDER = 6
L_ELBOW = 7
R_ELBOW = 8
L_WRIST = 9
R_WRIST = 10
L_HIP = 11
R_HIP = 12
TORSO_POINTS = [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]
WRIST = {"left": L_WRIST, "right": R_WRIST}
FIGHTER_IDS = ("A", "B")


@dataclass
class SparringConfig:
    min_kp_conf: float = 0.3
    max_gap_s: float = 0.25
    min_speed: float = 3.5
    prominence_ratio: float = 0.6
    min_gap_s: float = 0.18
    cross_hand_gap_s: float = 0.10
    jab_max_arc: float = 0.25
    span_back_s: float = 0.4
    span_forward_s: float = 0.25
    hit_margin: float = 0.35
    match_max_cost: float = 2.0
    color_weight: float = 0.0
    min_color_sep: float = 0.04
    learn_frames: int = 45
    min_reach: float = 0.65
    lum_weight: float = 0.0


@dataclass
class Detection:
    kp: np.ndarray
    conf: np.ndarray
    box: np.ndarray
    color: np.ndarray | None = None


@dataclass
class Track:
    kp: np.ndarray
    conf: np.ndarray
    box: np.ndarray


def torso_color(frame, kp, conf, min_conf=0.3):
    ok = conf[TORSO_POINTS] >= min_conf
    if int(ok.sum()) < 3:
        return None
    pts = kp[TORSO_POINTS][ok]
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    w, h = x1 - x0, y1 - y0
    x0, x1 = x0 + 0.2 * w, x1 - 0.2 * w
    y0, y1 = y0 + 0.2 * h, y1 - 0.2 * h
    height, width = frame.shape[:2]
    xs0, xs1 = int(max(0, x0)), int(min(width, x1))
    ys0, ys1 = int(max(0, y0)), int(min(height, y1))
    if xs1 - xs0 < 2 or ys1 - ys0 < 2:
        return None
    patch = frame[ys0:ys1, xs0:xs1]
    return patch.reshape(-1, 3).mean(axis=0) / 255.0


def appearance(frame, kp, conf, box=None, min_conf=0.3):
    feats = np.full(6, np.nan)
    torso = torso_color(frame, kp, conf, min_conf)
    if torso is not None:
        feats[:3] = torso
    height, width = frame.shape[:2]
    box_h = float(box[3] - box[1]) if box is not None else 300.0
    r = max(3, int(0.045 * box_h))
    patches = []
    for j in (L_WRIST, R_WRIST):
        if conf[j] < min_conf:
            continue
        x, y = int(kp[j][0]), int(kp[j][1])
        x0, x1 = max(0, x - r), min(width, x + r)
        y0, y1 = max(0, y - r), min(height, y + r)
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        patches.append(frame[y0:y1, x0:x1].reshape(-1, 3).mean(axis=0) / 255.0)
    if patches:
        feats[3:] = np.mean(patches, axis=0)
    return feats


def appearance_distance(a, b):
    if a is None or b is None:
        return None
    diff = (np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) ** 2
    ok = ~np.isnan(diff)
    if not ok.any():
        return None
    return float(np.sqrt(diff[ok].mean()))


def torso_brightness(det):
    color = det.color
    if color is None or np.isnan(np.asarray(color)[:3]).any():
        return None
    return float(np.mean(np.asarray(color)[:3]))


def learn_brightness_refs(frames, min_height=150, min_sep=0.08):
    """Two-cluster split of torso brightness over the whole clip: (dark, light) or None."""
    vals = np.array([v for dets in frames for d in dets
                     if (d.box[3] - d.box[1]) > min_height and (v := torso_brightness(d)) is not None])
    if len(vals) < 200:
        return None
    lo, hi = np.percentile(vals, 10), np.percentile(vals, 90)
    for _ in range(25):
        mid = (lo + hi) / 2
        a, b = vals[vals <= mid], vals[vals > mid]
        if len(a) == 0 or len(b) == 0:
            return None
        lo, hi = a.mean(), b.mean()
    return (float(lo), float(hi)) if hi - lo >= min_sep else None


def _center(box):
    return np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0])


def _area(box):
    return float((box[2] - box[0]) * (box[3] - box[1]))


def _usable(det, cfg):
    height = float(det.box[3] - det.box[1])
    return height > 0 and int((det.conf[TORSO_POINTS] >= cfg.min_kp_conf).sum()) >= 3


class FighterTracker:
    def __init__(self, fps, cfg=None, lum_refs=None):
        self.cfg = cfg or SparringConfig()
        self.fps = fps
        self.lum_refs = lum_refs  # (dark, light) torso brightness of the two fighters, or None
        self.lum_slots = None
        self.centers = [None, None]
        self.heights = [1.0, 1.0]
        self.missed = [0, 0]
        self.learn = [[], []]
        self.refs = None
        self.sep = 0.0

    def update(self, dets):
        usable = [d for d in dets if _usable(d, self.cfg)]
        if self.centers[0] is None:
            out = self._init(usable)
        else:
            out = self._match(usable)
        self._learn(out)
        return out

    def _set_state(self, slot, det):
        self.centers[slot] = _center(det.box)
        self.heights[slot] = float(det.box[3] - det.box[1])
        self.missed[slot] = 0

    def _assign_brightness(self, out):
        if self.lum_refs is None or self.lum_slots is not None or out[0] is None or out[1] is None:
            return
        if _iou(out[0].box, out[1].box) > 0.05:
            return
        lums = [torso_brightness(d) for d in out]
        if None in lums or abs(lums[0] - lums[1]) < 0.5 * (self.lum_refs[1] - self.lum_refs[0]):
            return
        dark, light = self.lum_refs
        self.lum_slots = (dark, light) if lums[0] < lums[1] else (light, dark)

    def _learn(self, out):
        self._assign_brightness(out)
        if self.refs is not None or out[0] is None or out[1] is None:
            return
        if _iou(out[0].box, out[1].box) > 0.05:
            return
        for slot in (0, 1):
            if out[slot].color is not None:
                self.learn[slot].append(np.asarray(out[slot].color, dtype=float))
        if min(len(self.learn[0]), len(self.learn[1])) >= self.cfg.learn_frames:
            refs = [np.nanmedian(np.stack(self.learn[slot]), axis=0) for slot in (0, 1)]
            sep = appearance_distance(refs[0], refs[1])
            self.refs = refs
            self.sep = sep or 0.0

    def _init(self, usable):
        if len(usable) < 2:
            return [None, None]
        ranked = sorted(usable, key=lambda d: _area(d.box), reverse=True)[:2]
        ranked.sort(key=lambda d: _center(d.box)[0])
        for slot, det in enumerate(ranked):
            self._set_state(slot, det)
        return ranked

    def _color_cost(self, det, slot):
        if self.lum_slots is not None:
            lum = torso_brightness(det)
            if lum is None:
                return 0.0
            span = abs(self.lum_refs[1] - self.lum_refs[0])
            return self.cfg.lum_weight * abs(lum - self.lum_slots[slot]) / span
        if self.refs is None or det.color is None or self.sep < self.cfg.min_color_sep:
            return 0.0
        dist = appearance_distance(det.color, self.refs[slot])
        if dist is None:
            return 0.0
        return self.cfg.color_weight * dist / self.sep

    def _match(self, usable):
        out = [None, None]
        if not usable:
            self.missed = [m + 1 for m in self.missed]
            return out
        cost = np.zeros((len(usable), 2))
        pos = np.zeros((len(usable), 2))
        for i, det in enumerate(usable):
            center = _center(det.box)
            for slot in (0, 1):
                dist = np.linalg.norm(center - self.centers[slot]) / max(self.heights[slot], 1.0)
                pos[i, slot] = dist
                cost[i, slot] = dist + self._color_cost(det, slot)
        rows, cols = linear_sum_assignment(cost)
        matched = set()
        for row, slot in zip(rows, cols):
            allowed = self.cfg.match_max_cost * (1.0 + self.missed[slot] / max(1.0, self.fps * 0.5))
            if pos[row, slot] <= allowed:
                out[slot] = usable[row]
                self._set_state(slot, usable[row])
                matched.add(slot)
        for slot in (0, 1):
            if slot not in matched:
                self.missed[slot] += 1
        return out


def resolve_identity(pairs, min_margin=0.06, min_len=5, overlap_iou=0.02):
    """Fix fighter swaps using shirt brightness. Slot 0 becomes the darker-shirted fighter.

    Frames are cut into stretches where both boxes are visible and apart, so the tracker could not
    have swapped inside them. A stretch whose two shirts clearly differ in brightness is forced into
    the dark-first order; unclear stretches keep whatever order the tracker had.
    """
    n = len(pairs)
    apart = np.zeros(n, bool)
    for k, p in enumerate(pairs):
        apart[k] = p[0] is not None and p[1] is not None and _iou(p[0].box, p[1].box) <= overlap_iou
    stretches, start = [], None
    for k in range(n + 1):
        ok = apart[k] if k < n else False
        if ok and start is None:
            start = k
        elif not ok and start is not None:
            if k - start >= min_len:
                stretches.append((start, k))
            start = None
    flips = []
    for s, e in stretches:
        med = []
        for slot in (0, 1):
            vals = [torso_brightness(pairs[k][slot]) for k in range(s, e)]
            vals = [v for v in vals if v is not None]
            med.append(float(np.median(vals)) if vals else None)
        if None in med or abs(med[0] - med[1]) < min_margin:
            flips.append(None)
        else:
            flips.append(med[0] > med[1])
    if all(f is None for f in flips):
        return pairs, 0
    current = next(f for f in flips if f is not None)
    state = np.zeros(n, bool)
    cursor = 0
    for (s, e), f in zip(stretches, flips):
        if f is not None:
            current = f
        state[cursor:e] = current
        cursor = e
    state[cursor:] = current
    out = [[p[1], p[0]] if state[k] else list(p) for k, p in enumerate(pairs)]
    changed = int(np.sum(state[1:] != state[:-1]))
    return out, changed


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def stack_tracks(pairs, num_kp=17):
    total = len(pairs)
    tracks = []
    for slot in (0, 1):
        kp = np.full((total, num_kp, 2), np.nan)
        conf = np.zeros((total, num_kp))
        box = np.full((total, 4), np.nan)
        for t, pair in enumerate(pairs):
            det = pair[slot]
            if det is None:
                continue
            kp[t] = det.kp
            conf[t] = det.conf
            box[t] = det.box
        tracks.append(Track(kp=kp, conf=conf, box=box))
    return tracks


def _interp_short_gaps(x, max_gap):
    out = x.copy()
    valid = ~np.isnan(x)
    if int(valid.sum()) < 2:
        return out
    idx = np.arange(len(x))
    filled = np.interp(idx, idx[valid], x[valid])
    n = len(x)
    i = 0
    while i < n:
        if valid[i]:
            i += 1
            continue
        j = i
        while j < n and not valid[j]:
            j += 1
        if i > 0 and j < n and (j - i) <= max_gap:
            out[i:j] = filled[i:j]
        i = j
    return out


def clean_track(track, fps, cfg):
    kp = track.kp.copy()
    kp[track.conf < cfg.min_kp_conf] = np.nan
    max_gap = max(1, int(round(cfg.max_gap_s * fps)))
    for j in range(kp.shape[1]):
        for c in range(2):
            kp[:, j, c] = _interp_short_gaps(kp[:, j, c], max_gap)
    kp = _nan_smooth(kp, size=3)
    return Track(kp=kp, conf=track.conf, box=track.box)


def _nan_smooth(kp, size):
    valid = ~np.isnan(kp)
    weights = np.ones(size) / size
    num = convolve1d(np.where(valid, kp, 0.0), weights, axis=0, mode="nearest")
    den = convolve1d(valid.astype(float), weights, axis=0, mode="nearest")
    out = np.full(kp.shape, np.nan)
    ok = valid & (den > 1e-9)
    out[ok] = num[ok] / den[ok]
    return out


def _mid_shoulder(kp):
    return (kp[:, L_SHOULDER] + kp[:, R_SHOULDER]) / 2.0


def _head(kp):
    return np.where(np.isnan(kp[:, NOSE]), _mid_shoulder(kp), kp[:, NOSE])


def _ffill(arr):
    out = arr.copy()
    for c in range(out.shape[1]):
        col = out[:, c]
        valid = ~np.isnan(col)
        if not valid.any():
            continue
        idx = np.where(valid, np.arange(len(col)), 0)
        np.maximum.accumulate(idx, out=idx)
        filled = col[idx]
        filled[: int(np.argmax(valid))] = col[int(np.argmax(valid))]
        out[:, c] = filled
    return out


def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-6)


def torso_length(track):
    kp = track.kp
    mid_hip = (kp[:, L_HIP] + kp[:, R_HIP]) / 2.0
    d = np.linalg.norm(_mid_shoulder(kp) - mid_hip, axis=1)
    d = d[~np.isnan(d)]
    if len(d) >= 5:
        return float(np.median(d))
    heights = track.box[:, 3] - track.box[:, 1]
    heights = heights[~np.isnan(heights)]
    return float(np.median(heights) * 0.3) if len(heights) else 1.0


def lead_side(me, opp, torso):
    mid = _mid_shoulder(me.kp)
    u = _unit(_ffill(_head(opp.kp)) - mid)
    scores = {}
    for side, j in WRIST.items():
        proj = np.sum((me.kp[:, j] - mid) * u, axis=1) / torso
        scores[side] = float(np.nanmedian(proj)) if np.isfinite(proj).any() else -np.inf
    return "left" if scores["left"] >= scores["right"] else "right"


def lead_series(me, opp, torso, fps, smooth_s=2.0, deadband=0.04):
    mid = _mid_shoulder(me.kp)
    u = _unit(_ffill(_head(opp.kp)) - mid)
    proj = {
        side: np.sum((me.kp[:, j] - mid) * u, axis=1) / torso for side, j in WRIST.items()
    }
    diff = _ffill((proj["left"] - proj["right"]).reshape(-1, 1))[:, 0]
    diff = np.nan_to_num(diff, nan=0.0)
    size = max(3, int(round(smooth_s * fps)) | 1)
    smooth = median_filter(diff, size=size, mode="nearest")
    lead_left = np.empty(len(smooth), dtype=bool)
    current = bool(smooth[0] >= 0)
    for k, value in enumerate(smooth):
        if current and value < -deadband:
            current = False
        elif not current and value > deadband:
            current = True
        lead_left[k] = current
    return lead_left


def _detect(me, opp, side, torso, fps, cfg):
    j = WRIST[side]
    mid = _mid_shoulder(me.kp)
    rel = (me.kp[:, j] - mid) / torso
    vel = np.gradient(rel, 1.0 / fps, axis=0)
    u = _unit(_ffill(_head(opp.kp)) - mid)
    approach = np.nan_to_num(np.sum(vel * u, axis=1), nan=0.0)
    peaks, _ = find_peaks(
        approach,
        height=cfg.min_speed,
        distance=max(1, int(round(cfg.min_gap_s * fps))),
        prominence=cfg.min_speed * cfg.prominence_ratio,
    )
    return peaks, approach, rel


def _movement_span(approach, rel, peak, fps, cfg):
    floor = 0.15 * approach[peak]
    limit = max(0, peak - int(round(cfg.span_back_s * fps)))
    lo = peak
    while lo > limit and approach[lo - 1] > floor:
        lo -= 1
    lo = max(0, lo - 1)
    top = min(len(rel) - 1, peak + int(round(cfg.span_forward_s * fps)))
    reach = np.linalg.norm(rel[peak : top + 1], axis=1)
    hi = peak + int(np.nanargmax(reach)) if np.isfinite(reach).any() else peak
    return lo, hi


def _arc_ratio(rel, lo, hi):
    seg = rel[lo : hi + 1]
    seg = seg[~np.isnan(seg).any(axis=1)]
    if len(seg) < 3:
        return 0.0
    chord = seg[-1] - seg[0]
    length = float(np.linalg.norm(chord))
    if length < 1e-6:
        return 1.0
    d = chord / length
    offsets = (seg[:, 0] - seg[0, 0]) * d[1] - (seg[:, 1] - seg[0, 1]) * d[0]
    return float(np.abs(offsets).max() / length)


def _reach(me, side, lo, hi, torso):
    shoulder = L_SHOULDER if side == "left" else R_SHOULDER
    seg = me.kp[lo : hi + 1, WRIST[side]] - me.kp[lo : hi + 1, shoulder]
    d = np.linalg.norm(seg, axis=1) / torso
    return float(np.nanmax(d)) if np.isfinite(d).any() else 0.0


def _landed(opp, opp_torso, wrist_xy, lo, hi, cfg):
    margin = cfg.hit_margin * opp_torso
    for k in range(lo, min(len(wrist_xy), hi + 1)):
        pts = opp.kp[k][[NOSE] + TORSO_POINTS]
        pts = pts[~np.isnan(pts).any(axis=1)]
        w = wrist_xy[k]
        if len(pts) < 2 or np.isnan(w).any():
            continue
        if np.all(w >= pts.min(axis=0) - margin) and np.all(w <= pts.max(axis=0) + margin):
            return True
    return False


def _dedupe(events, gap_s):
    keep = []
    for event in sorted(events, key=lambda e: e["t"]):
        if keep and event["t"] - keep[-1]["t"] < gap_s:
            if event["speed"] > keep[-1]["speed"]:
                keep[-1] = event
            continue
        keep.append(event)
    return keep


def analyze_full(pairs, fps, cfg=None, start_s=0.0):
    cfg = cfg or SparringConfig()
    if len(pairs) < 3:
        return {"punches": [], "stances": ["orthodox", "orthodox"], "tracks": None, "torsos": None, "leads": None}
    tracks = [clean_track(t, fps, cfg) for t in stack_tracks(pairs)]
    torsos = [torso_length(t) for t in tracks]
    leads = [lead_series(tracks[i], tracks[1 - i], torsos[i], fps) for i in (0, 1)]
    stances = ["orthodox" if float(leads[i].mean()) >= 0.5 else "southpaw" for i in (0, 1)]
    punches = []
    for i in (0, 1):
        me, opp = tracks[i], tracks[1 - i]
        events = []
        for side in ("left", "right"):
            peaks, approach, rel = _detect(me, opp, side, torsos[i], fps, cfg)
            wrist_xy = me.kp[:, WRIST[side]]
            for peak in peaks:
                peak = int(peak)
                lo, hi = _movement_span(approach, rel, peak, fps, cfg)
                reach = _reach(me, side, lo, hi, torsos[i])
                if reach < cfg.min_reach:
                    continue
                is_lead = (side == "left") == bool(leads[i][peak])
                is_jab = is_lead and _arc_ratio(rel, lo, hi) <= cfg.jab_max_arc
                events.append(
                    {
                        "t": round(start_s + float(peak) / fps, 3),
                        "fighter": FIGHTER_IDS[i],
                        "hand": side,
                        "lead": bool(is_lead),
                        "type": "jab" if is_jab else "power",
                        "landed": bool(_landed(opp, torsos[1 - i], wrist_xy, peak, hi, cfg)),
                        "speed": round(float(approach[peak]), 1),
                        "reach": round(reach, 2),
                    }
                )
        punches.extend(_dedupe(events, cfg.cross_hand_gap_s))
    punches.sort(key=lambda e: e["t"])
    return {"punches": punches, "stances": stances, "tracks": tracks, "torsos": torsos, "leads": leads}


def analyze(pairs, fps, cfg=None, start_s=0.0):
    result = analyze_full(pairs, fps, cfg, start_s)
    return result["punches"], result["stances"]


def summarize(punches):
    counts = {}
    for fid in FIGHTER_IDS:
        mine = [p for p in punches if p["fighter"] == fid]
        counts[fid] = {
            "thrown": len(mine),
            "landed": sum(1 for p in mine if p["landed"]),
            "jab": sum(1 for p in mine if p["type"] == "jab"),
            "power": sum(1 for p in mine if p["type"] == "power"),
        }
    return counts


def build_report(punches, stances, labels, fps, start_s, end_s, meta):
    return {
        "version": 1,
        **meta,
        "fps": round(float(fps), 3),
        "start": round(float(start_s), 3),
        "end": round(float(end_s), 3),
        "fighters": [
            {"id": FIGHTER_IDS[i], "label": labels[i], "stance": stances[i]} for i in (0, 1)
        ],
        "punches": punches,
        "counts": summarize(punches),
    }
