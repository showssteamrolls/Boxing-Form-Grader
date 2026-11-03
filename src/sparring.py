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
L_EYE, R_EYE, L_EAR, R_EAR = 1, 2, 3, 4
FACE_POINTS = [NOSE, L_EYE, R_EYE, L_EAR, R_EAR]
TORSO_POINTS = [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]
WRIST = {"left": L_WRIST, "right": R_WRIST}
ELBOW = {"left": L_ELBOW, "right": R_ELBOW}
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
    head_radius: float = 0.22      # head target, in opponent torso lengths around the nose
    mutual_gap_s: float = 0.30      # both fighters landing inside this window: only the faster punch keeps it
    depth_tol: float = 0.6         # max feet-line gap, in torso lengths, for two fighters to count as in reach
    glove_clear: float = 0.10      # contacts this close to the opponent's glove count as parries
    arm_clear: float = 0.06        # contacts this close to an upper arm or forearm count as arm hits
    match_max_cost: float = 2.0
    color_weight: float = 0.0
    min_color_sep: float = 0.04
    learn_frames: int = 45
    min_reach: float = 0.65
    lum_weight: float = 0.0
    contact_slack_s: float = 0.12  # keep checking this long past full extension; the pose lags the glove
    min_height_share: float = 0.5  # a detection shorter than this share of the tallest one is not a fighter
    min_extension: float = 0.20    # the hand must travel at least this far out (torso lengths) during the punch
    burst_ratio: float = 1.5       # peak speed over the median hand speed in the second before it; waving has no burst
    glove_ext: float = 0.30        # the glove reaches this far past the wrist along the forearm, in torso lengths
    zone_pad: float = 0.20         # glove radius added around each body zone, in torso lengths


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


APPEARANCE_DIMS = 7  # torso BGR, wrist-patch BGR, red-glove share


def appearance(frame, kp, conf, box=None, min_conf=0.3):
    feats = np.full(APPEARANCE_DIMS, np.nan)
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
        feats[3:6] = np.mean(patches, axis=0)
    reds = []
    big = max(4, int(0.09 * box_h))
    for j in (L_WRIST, R_WRIST):
        if conf[j] < min_conf:
            continue
        x, y = int(kp[j][0]), int(kp[j][1])
        patch = frame[max(0, y - big): y + big, max(0, x - big): x + big].astype(float)
        if patch.shape[0] < 3 or patch.shape[1] < 3:
            continue
        b, g, r_ = patch[..., 0], patch[..., 1], patch[..., 2]
        reds.append(float(np.mean((r_ > 90) & (r_ > 1.7 * g) & (r_ > 1.7 * b))))
    if reds:
        feats[6] = float(np.mean(reds))  # share of red glove pixels around the wrists
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
        if usable:
            # onlookers, and people stretching on the floor, are much shorter in the frame than the fighters
            tallest = max(float(d.box[3] - d.box[1]) for d in usable)
            usable = [d for d in usable if float(d.box[3] - d.box[1]) >= self.cfg.min_height_share * tallest]
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


def identity_score(det):
    """Higher = more like the dark-topped, red-gloved fighter. None when nothing is measurable."""
    if det is None or det.color is None:
        return None
    c = np.asarray(det.color, dtype=float)
    red = c[6] if len(c) > 6 and not np.isnan(c[6]) else None
    lum = torso_brightness(det)
    if red is None and lum is None:
        return None
    return (red if red is not None else 0.0) + (2.0 * (0.25 - lum) if lum is not None else 0.0)


def resolve_identity(pairs, min_margin=0.08, window=15):
    """Fix fighter swaps from appearance. Slot 0 becomes the fighter with the redder gloves and darker top.

    Every frame with both fighters gets a margin: how much more "red glove, dark top" slot 0 looks
    than slot 1. The margin is median-smoothed over about half a second. Where it is clearly
    positive the order is kept, where it is clearly negative the two are swapped, and where it is
    unclear the previous frame's decision holds. This also repairs swaps inside clinches.
    """
    n = len(pairs)
    margin = np.full(n, np.nan)
    for k, p in enumerate(pairs):
        a, b = identity_score(p[0]), identity_score(p[1])
        if a is not None and b is not None:
            margin[k] = a - b
    if not np.isfinite(margin).any():
        return pairs, 0
    half = window // 2
    smooth = np.full(n, np.nan)
    for k in range(n):
        w = margin[max(0, k - half): k + half + 1]
        w = w[np.isfinite(w)]
        if len(w) >= 3:
            smooth[k] = np.median(w)
    state = np.zeros(n, bool)
    current = None
    first = next((k for k in range(n) if np.isfinite(smooth[k]) and abs(smooth[k]) >= min_margin), None)
    if first is None:
        return pairs, 0
    current = smooth[first] < 0
    for k in range(n):
        if np.isfinite(smooth[k]) and abs(smooth[k]) >= min_margin:
            current = smooth[k] < 0
        state[k] = current
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


def _is_burst(approach, rel, lo, hi, peak, fps, cfg):
    """False for a hand that drifts out or waves: it has to travel out and burst past its own baseline."""
    r = np.linalg.norm(rel[[lo, hi]], axis=1)
    if np.isfinite(r).all() and r[1] - r[0] < cfg.min_extension:
        return False
    before = np.abs(approach[max(0, lo - int(round(fps))) : lo])
    if len(before) >= 3 and approach[peak] < cfg.burst_ratio * float(np.median(before)):
        return False
    return True


def _reach(me, side, lo, hi, torso):
    shoulder = L_SHOULDER if side == "left" else R_SHOULDER
    seg = me.kp[lo : hi + 1, WRIST[side]] - me.kp[lo : hi + 1, shoulder]
    d = np.linalg.norm(seg, axis=1) / torso
    return float(np.nanmax(d)) if np.isfinite(d).any() else 0.0


def head_center(kp, torso):
    """Head position: the nose when it sits where a head should, otherwise up the torso axis.

    A glove over the face makes the nose drop out or jump, so the axis estimate keeps the head
    target in place. The target's size is fixed (a fraction of torso length), never read from keypoints.
    """
    sh = kp[[L_SHOULDER, R_SHOULDER]]
    hp = kp[[L_HIP, R_HIP]]
    axis = None
    if not np.isnan(sh).any() and not np.isnan(hp).any():
        up = sh.mean(axis=0) - hp.mean(axis=0)
        norm = float(np.linalg.norm(up))
        if norm > 1e-6:
            axis = sh.mean(axis=0) + up / norm * 0.45 * torso
    nose = kp[NOSE]
    if not np.isnan(nose).any():
        if axis is None or np.linalg.norm(nose - axis) <= 0.6 * torso:
            return nose
    return axis


HEAD_W = 0.42   # upright head width, in torso lengths
HEAD_H = 0.52   # upright head height (forehead to chin), in torso lengths


def _torso_frame(kp):
    sh = kp[[L_SHOULDER, R_SHOULDER]]
    hp = kp[[L_HIP, R_HIP]]
    if np.isnan(sh).any() or np.isnan(hp).any():
        return None
    mid = sh.mean(axis=0)
    up = mid - hp.mean(axis=0)
    norm = float(np.linalg.norm(up))
    if norm < 1e-6:
        return None
    return mid, up / norm


def head_rise(kp, torso):
    """How high the nose sits above the shoulders, in torso lengths. Drops when the chin tucks."""
    frame = _torso_frame(kp)
    nose = kp[NOSE]
    if frame is None or np.isnan(nose).any():
        return np.nan
    mid, up = frame
    return float((nose - mid) @ up) / torso


def head_ellipse(kp, torso, ref_rise):
    """Face target (cx, cy, half_width, half_height) in pixels, or None without a torso.

    Built from the nose, eyes and ears: it spans forehead, eyes and chin across to the ears. Chin
    down brings the nose closer to the shoulders, which shrinks the height. Points that sit
    far from where a head can be (a glove mistaken for an ear) are ignored.
    """
    frame = _torso_frame(kp)
    if frame is None:
        return None
    mid, up = frame
    anchor = mid + up * 0.45 * torso
    pts = {}
    for name, j in (("nose", NOSE), ("le", L_EYE), ("re", R_EYE), ("lr", L_EAR), ("rr", R_EAR)):
        p = kp[j]
        if not np.isnan(p).any() and np.linalg.norm(p - anchor) <= 0.6 * torso:
            pts[name] = p
    rise = head_rise(kp, torso) if "nose" in pts else np.nan
    scale = 1.0 if np.isnan(rise) else float(np.clip(rise / ref_rise, 0.65, 1.0))
    height = HEAD_H * torso * scale
    width = HEAD_W * torso
    if "lr" in pts and "rr" in pts:
        width = float(np.clip(np.linalg.norm(pts["lr"] - pts["rr"]) * 1.25, 0.55 * width, 1.15 * width))
    elif "le" in pts and "re" in pts:
        width = float(np.clip(np.linalg.norm(pts["le"] - pts["re"]) * 2.5, 0.55 * width, 1.15 * width))
    else:
        width *= 0.8
    eyes = [pts[k] for k in ("le", "re") if k in pts]
    if eyes:
        eye_y = float(np.mean([e[1] for e in eyes]))
    elif "nose" in pts:
        eye_y = float(pts["nose"][1] - 0.2 * height)
    else:
        eye_y = float(anchor[1])
    cx = float(np.mean([p[0] for p in pts.values()])) if pts else float(anchor[0])
    cy = eye_y + 0.05 * height
    return cx, cy, width / 2.0, height / 2.0


def head_series(track, torso):
    """Per-frame face ellipses (n, 4), smoothed, with short gaps held. NaN where nothing is known."""
    kp = track.kp
    rises = np.array([head_rise(kp[k], torso) for k in range(len(kp))])
    ref = float(np.clip(np.nanpercentile(rises, 85), 0.3, 0.8)) if np.isfinite(rises).sum() > 10 else 0.5
    out = np.full((len(kp), 4), np.nan)
    for k in range(len(kp)):
        e = head_ellipse(kp[k], torso, ref)
        if e is not None:
            out[k] = e
    out = _nan_smooth(out, size=5)
    for c in range(4):
        out[:, c] = _interp_short_gaps(out[:, c], 15)
    return out


# Body targets, in the torso's own coordinates: psi is the angle around the body (0 = front center,
# + = the fighter's right), b runs from the hip line (0) to the shoulder line (1), wz is half-width as a
# share of body half-width, hb is half-height in torso lengths. Chest is left out on purpose.
ZONES = (
    ("solar_plexus", 0.0, 0.66, 0.30, 0.12),
    ("liver", 0.50, 0.50, 0.32, 0.14),
    ("spleen", -0.58, 0.52, 0.30, 0.14),
    ("core", 0.0, 0.28, 0.40, 0.16),
)
ZONE_NAMES = tuple(z[0] for z in ZONES)
ZONE_COLS = 6   # cx, cy, rx, ry, tilt (radians), hittable (0 or 1)


def _side_view(kp, torso, refs):
    """Signed cos and sin of the body's turn away from facing the camera, one value per frame.

    cos comes from how far apart the shoulders sit (left minus right, so a back turned to the camera
    goes negative), against the widest spread seen for that fighter. The sign of sin comes from which
    way the nose points off the shoulder line. Face points that are visible mean the chest faces the
    camera, even if the pose model swapped left and right.
    """
    n = len(kp)
    sh_ref, _ = refs
    spread = (kp[:, L_SHOULDER, 0] - kp[:, R_SHOULDER, 0]) / max(sh_ref * torso, 1e-6)
    face_seen = ~np.isnan(kp[:, NOSE, 0]) & (~np.isnan(kp[:, L_EYE, 0]) | ~np.isnan(kp[:, R_EYE, 0]))
    back_seen = np.isnan(kp[:, NOSE, 0]) & ~np.isnan(kp[:, L_EAR, 0]) & ~np.isnan(kp[:, R_EAR, 0])
    cos = np.clip(spread, -1.0, 1.0)
    cos = np.where(face_seen & (cos < 0), np.abs(cos), cos)
    cos = np.where(back_seen & (cos > 0), -np.abs(cos), cos)
    cos = _nan_smooth(cos[:, None], 5)[:, 0]
    mid = _mid_shoulder(kp)
    offset = (kp[:, NOSE, 0] - mid[:, 0]) / max(torso, 1e-6)
    sign = np.where(np.abs(offset) > 0.06, np.sign(offset), np.nan)
    return cos, sign


def zone_series(track, torso, opp_track):
    """Per-frame body targets, shape (n, len(ZONES), 6). NaN where the torso is unknown.

    Each zone is a point on the surface of an elliptical torso (width from the shoulder and hip
    spread, depth 0.6 of the width), turned by the body's estimated yaw, so it follows the body when
    it rotates and slides toward the near edge in profile. 'Hittable' means that surface point faces
    the opponent or the camera; a back turned to both is not a target.
    """
    kp = track.kp
    n = len(kp)
    out = np.full((n, len(ZONES), ZONE_COLS), np.nan)
    if n == 0:
        return out
    sh_w = np.abs(kp[:, L_SHOULDER, 0] - kp[:, R_SHOULDER, 0]) / max(torso, 1e-6)
    hp_w = np.abs(kp[:, L_HIP, 0] - kp[:, R_HIP, 0]) / max(torso, 1e-6)
    sh_ref = float(np.clip(np.nanpercentile(sh_w, 90), 0.45, 1.2)) if np.isfinite(sh_w).sum() > 10 else 0.8
    hp_ref = float(np.clip(np.nanpercentile(hp_w, 90), 0.3, 1.0)) if np.isfinite(hp_w).sum() > 10 else 0.6
    cos, sign = _side_view(kp, torso, (sh_ref, hp_ref))
    opp_mid = _mid_shoulder(opp_track.kp)
    my_mid = _mid_shoulder(kp)
    toward = np.sign(opp_mid[:, 0] - my_mid[:, 0])
    toward = _nan_smooth(toward[:, None], 15)[:, 0]
    toward = np.where(np.abs(toward) < 0.1, 1.0, np.sign(toward))
    sign = np.where(np.isnan(sign), toward, sign)
    sin = _nan_smooth((sign * np.sqrt(np.clip(1 - cos**2, 0, 1)))[:, None], 7)[:, 0]
    phi = np.arctan2(sin, cos)
    shoulders = _mid_shoulder(kp)
    hips = (kp[:, L_HIP] + kp[:, R_HIP]) / 2.0
    up = shoulders - hips
    tilt = np.arctan2(up[:, 0], -up[:, 1])
    for k in range(n):
        if np.isnan(up[k]).any() or np.isnan(phi[k]):
            continue
        length = float(np.linalg.norm(up[k]))
        for zi, (_, psi, b, wz, hb) in enumerate(ZONES):
            half = (hp_ref + (sh_ref - hp_ref) * b) * torso / 2.0
            depth = 0.7 * half
            cx = hips[k, 0] + b * up[k, 0] + half * np.sin(psi) * -np.cos(phi[k]) + depth * np.cos(psi) * np.sin(phi[k])
            cy = hips[k, 1] + b * up[k, 1]
            rx = max(0.18 * torso, wz * (half * abs(np.cos(phi[k])) + depth * abs(np.sin(phi[k]))))
            ry = hb * length
            chi = phi[k] - psi
            hittable = (toward[k] * np.sin(chi) > 0.5) or (np.cos(chi) > 0.6)
            out[k, zi] = (cx, cy, rx, ry, tilt[k], 1.0 if hittable else 0.0)
    for zi in range(len(ZONES)):
        for c in range(5):
            out[:, zi, c] = _nan_smooth(out[:, zi, c][:, None], 5)[:, 0]
            out[:, zi, c] = _interp_short_gaps(out[:, zi, c], 15)
        out[:, zi, 5] = _interp_short_gaps(out[:, zi, 5], 15)
    return out


def _in_zone(w, zone, pad):
    cx, cy, rx, ry, tilt, _ = zone
    dx, dy = w[0] - cx, w[1] - cy
    c, s = np.cos(tilt), np.sin(tilt)
    u = dx * c + dy * s        # along the zone's width axis
    v = -dx * s + dy * c       # along its height axis
    return (u / (rx + pad)) ** 2 + (v / (ry + pad)) ** 2 <= 1.0



def _seg_dist(p, a, b):
    ab = b - a
    denom = float(ab @ ab)
    t = 0.0 if denom < 1e-9 else float(np.clip((p - a) @ ab / denom, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def _in_quad(p, quad):
    sign = 0
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        if abs(cross) < 1e-9:
            continue
        cur = 1 if cross > 0 else -1
        if sign == 0:
            sign = cur
        elif cur != sign:
            return False
    return True


def _same_depth(me, opp, lo, hi, torso, cfg):
    """Fighters standing at very different distances from the camera can overlap in 2D without touching.

    The bottom of each fighter's box tracks where their feet meet the floor, so a large gap means
    one is much deeper in the scene. Unknown (no boxes) passes.
    """
    a = me.box[lo : hi + 1, 3]
    b = opp.box[lo : hi + 1, 3]
    gap = np.abs(a - b) / max(torso, 1e-6)
    gap = gap[~np.isnan(gap)]
    if len(gap) == 0:
        return True
    return float(np.median(gap)) <= cfg.depth_tol


def _on_guard(kp, w, torso, cfg):
    """True if the contact point sits on the opponent's glove or arm (a parry or block)."""
    for wr in (L_WRIST, R_WRIST):
        p = kp[wr]
        if not np.isnan(p).any() and np.linalg.norm(w - p) <= cfg.glove_clear * torso:
            return True
    for el, wr in ((L_ELBOW, L_WRIST), (R_ELBOW, R_WRIST)):
        pa, pb = kp[el], kp[wr]
        if np.isnan(pa).any() or np.isnan(pb).any():
            continue
        if _seg_dist(w, pa, pb) <= cfg.arm_clear * torso:
            return True
    return False


def _landed(opp, opp_torso, wrist_xy, lo, hi, cfg, heads=None, zones=None):
    """Name of the target the wrist reaches (face or a body zone), else None. Falsy when nothing is hit.

    Face: the ellipse from forehead and eyes down to the chin, across to the ears. Body: the
    solar plexus, liver, spleen and core zones that face the attacker (or the camera). A contact is
    dropped when it sits on the opponent's glove or arm, so parries and arm hits do not count.
    """
    for k in range(lo, min(len(wrist_xy), hi + 1)):
        w = wrist_xy[k]
        if np.isnan(w).any():
            continue
        kp = opp.kp[k]
        if _on_guard(kp, w, opp_torso, cfg):
            continue
        if heads is not None and k < len(heads) and np.isfinite(heads[k]).all():
            cx, cy, rx, ry = heads[k]
            if ((w[0] - cx) / rx) ** 2 + ((w[1] - cy) / ry) ** 2 <= 1.0:
                return "head"
        else:
            head = head_center(kp, opp_torso)
            if head is not None and np.linalg.norm(w - head) <= cfg.head_radius * opp_torso:
                return "head"
        if zones is not None and k < len(zones):
            for zi, name in enumerate(ZONE_NAMES):
                z = zones[k, zi]
                if np.isfinite(z).all() and z[5] > 0.5 and _in_zone(w, z, cfg.zone_pad * opp_torso):
                    return name
    return None


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
        return {"punches": [], "stances": ["orthodox", "orthodox"], "tracks": None, "torsos": None, "leads": None, "heads": None, "zones": None}
    tracks = [clean_track(t, fps, cfg) for t in stack_tracks(pairs)]
    torsos = [torso_length(t) for t in tracks]
    leads = [lead_series(tracks[i], tracks[1 - i], torsos[i], fps) for i in (0, 1)]
    heads = [head_series(tracks[i], torsos[i]) for i in (0, 1)]
    zones = [zone_series(tracks[i], torsos[i], tracks[1 - i]) for i in (0, 1)]
    stances = ["orthodox" if float(leads[i].mean()) >= 0.5 else "southpaw" for i in (0, 1)]
    punches = []
    for i in (0, 1):
        me, opp = tracks[i], tracks[1 - i]
        events = []
        for side in ("left", "right"):
            peaks, approach, rel = _detect(me, opp, side, torsos[i], fps, cfg)
            wrist_xy = me.kp[:, WRIST[side]]
            elbow_xy = me.kp[:, ELBOW[side]]
            fore = wrist_xy - elbow_xy
            norm = np.linalg.norm(fore, axis=1, keepdims=True)
            tip_xy = np.where(np.isnan(fore).any(axis=1, keepdims=True) | (norm < 1e-6), wrist_xy,
                              wrist_xy + fore / np.maximum(norm, 1e-6) * cfg.glove_ext * torsos[i])
            for peak in peaks:
                peak = int(peak)
                lo, hi = _movement_span(approach, rel, peak, fps, cfg)
                reach = _reach(me, side, lo, hi, torsos[i])
                if reach < cfg.min_reach:
                    continue
                if not _is_burst(approach, rel, lo, hi, peak, fps, cfg):
                    continue
                is_lead = (side == "left") == bool(leads[i][peak])
                is_jab = is_lead and _arc_ratio(rel, lo, hi) <= cfg.jab_max_arc
                in_depth = _same_depth(me, opp, peak, hi, torsos[i], cfg)
                target = _landed(opp, torsos[1 - i], tip_xy, max(lo, peak - int(round(0.06 * fps))), hi + int(round(cfg.contact_slack_s * fps)), cfg, heads[1 - i], zones[1 - i]) if in_depth else None
                events.append(
                    {
                        "t": round(start_s + float(peak) / fps, 3),
                        "fighter": FIGHTER_IDS[i],
                        "hand": side,
                        "lead": bool(is_lead),
                        "type": "jab" if is_jab else "power",
                        "landed": target is not None,
                        "target": target,
                        "speed": round(float(approach[peak]), 1),
                        "reach": round(reach, 2),
                    }
                )
        punches.extend(_dedupe(events, cfg.cross_hand_gap_s))
    punches.sort(key=lambda e: e["t"])
    _resolve_mutual_landings(punches, cfg.mutual_gap_s)
    return {"punches": punches, "stances": stances, "tracks": tracks, "torsos": torsos, "leads": leads, "heads": heads, "zones": zones}


def _resolve_mutual_landings(punches, gap_s):
    """When both fighters register a landed punch at once, the harder-moving one is the thrower.

    The other glove is usually a guard or counter moving through the clinch, not a clean hit.
    """
    landed = [p for p in punches if p["landed"]]
    for p in landed:
        for q in landed:
            if q["fighter"] != p["fighter"] and abs(q["t"] - p["t"]) <= gap_s and p["landed"] and q["landed"]:
                weaker = p if p["speed"] < q["speed"] else q
                weaker["landed"] = False
                weaker["target"] = None


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
