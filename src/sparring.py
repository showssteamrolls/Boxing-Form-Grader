from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks

NOSE = 0
L_SHOULDER = 5
R_SHOULDER = 6
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
    min_speed: float = 5.0
    prominence_ratio: float = 0.6
    min_gap_s: float = 0.18
    cross_hand_gap_s: float = 0.10
    jab_max_arc: float = 0.25
    span_back_s: float = 0.4
    span_forward_s: float = 0.25
    hit_margin: float = 0.35
    match_max_cost: float = 2.0
    color_weight: float = 4.0


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


def _center(box):
    return np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0])


def _area(box):
    return float((box[2] - box[0]) * (box[3] - box[1]))


def _usable(det, cfg):
    height = float(det.box[3] - det.box[1])
    return height > 0 and int((det.conf[TORSO_POINTS] >= cfg.min_kp_conf).sum()) >= 3


class FighterTracker:
    def __init__(self, fps, cfg=None):
        self.cfg = cfg or SparringConfig()
        self.fps = fps
        self.centers = [None, None]
        self.heights = [1.0, 1.0]
        self.colors = [None, None]
        self.missed = [0, 0]

    def update(self, dets):
        usable = [d for d in dets if _usable(d, self.cfg)]
        if self.centers[0] is None:
            return self._init(usable)
        return self._match(usable)

    def _set_state(self, slot, det):
        self.centers[slot] = _center(det.box)
        self.heights[slot] = float(det.box[3] - det.box[1])
        self.missed[slot] = 0
        if det.color is not None:
            if self.colors[slot] is None:
                self.colors[slot] = np.array(det.color, dtype=float)
            else:
                self.colors[slot] = 0.9 * self.colors[slot] + 0.1 * det.color

    def _init(self, usable):
        if len(usable) < 2:
            return [None, None]
        ranked = sorted(usable, key=lambda d: _area(d.box), reverse=True)[:2]
        ranked.sort(key=lambda d: _center(d.box)[0])
        for slot, det in enumerate(ranked):
            self._set_state(slot, det)
        return ranked

    def _match(self, usable):
        out = [None, None]
        if not usable:
            self.missed = [m + 1 for m in self.missed]
            return out
        cost = np.zeros((len(usable), 2))
        for i, det in enumerate(usable):
            center = _center(det.box)
            for slot in (0, 1):
                dist = np.linalg.norm(center - self.centers[slot]) / max(self.heights[slot], 1.0)
                color = 0.0
                if det.color is not None and self.colors[slot] is not None:
                    color = self.cfg.color_weight * float(np.linalg.norm(det.color - self.colors[slot]))
                cost[i, slot] = dist + color
        rows, cols = linear_sum_assignment(cost)
        matched = set()
        for row, slot in zip(rows, cols):
            allowed = self.cfg.match_max_cost * (1.0 + self.missed[slot] / max(1.0, self.fps * 0.5))
            if cost[row, slot] <= allowed:
                out[slot] = usable[row]
                self._set_state(slot, usable[row])
                matched.add(slot)
        for slot in (0, 1):
            if slot not in matched:
                self.missed[slot] += 1
        return out


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
    kp = uniform_filter1d(kp, size=3, axis=0, mode="nearest")
    return Track(kp=kp, conf=track.conf, box=track.box)


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


def analyze(pairs, fps, cfg=None, start_s=0.0):
    cfg = cfg or SparringConfig()
    if len(pairs) < 3:
        return [], ["orthodox", "orthodox"]
    tracks = [clean_track(t, fps, cfg) for t in stack_tracks(pairs)]
    torsos = [torso_length(t) for t in tracks]
    punches = []
    stances = []
    for i in (0, 1):
        me, opp = tracks[i], tracks[1 - i]
        lead = lead_side(me, opp, torsos[i])
        stances.append("orthodox" if lead == "left" else "southpaw")
        events = []
        for side in ("left", "right"):
            peaks, approach, rel = _detect(me, opp, side, torsos[i], fps, cfg)
            wrist_xy = me.kp[:, WRIST[side]]
            for peak in peaks:
                peak = int(peak)
                lo, hi = _movement_span(approach, rel, peak, fps, cfg)
                is_jab = side == lead and _arc_ratio(rel, lo, hi) <= cfg.jab_max_arc
                events.append(
                    {
                        "t": round(start_s + float(peak) / fps, 3),
                        "fighter": FIGHTER_IDS[i],
                        "hand": side,
                        "type": "jab" if is_jab else "power",
                        "landed": bool(_landed(opp, torsos[1 - i], wrist_xy, peak, hi, cfg)),
                        "speed": round(float(approach[peak]), 1),
                    }
                )
        punches.extend(_dedupe(events, cfg.cross_hand_gap_s))
    punches.sort(key=lambda e: e["t"])
    return punches, stances


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
