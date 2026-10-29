from __future__ import annotations

import numpy as np

from sparring import (
    FIGHTER_IDS,
    L_ELBOW,
    L_HIP,
    L_SHOULDER,
    L_WRIST,
    R_ELBOW,
    R_HIP,
    R_SHOULDER,
    R_WRIST,
)

SERIES_DT = 0.25
POSE_HZ = 15
POSE_JOINTS = [0, 5, 6, 7, 8, 9, 10, 11, 12]

WINDOWS = {
    "output_s": 10.0,
    "land_s": 30.0,
    "lead_act_s": 5.0,
    "guard_s": 10.0,
    "pressure_s": 3.0,
    "angle_s": 0.5,
    "dist_s": 0.5,
    "speed_n": 8,
}
MIN_LAND_SAMPLE = 3
MIN_OUTPUT_WINDOW_S = 5.0


def trailing_mean(x, win):
    win = max(1, int(win))
    valid = ~np.isnan(x)
    filled = np.where(valid, x, 0.0)
    csum = np.concatenate([[0.0], np.cumsum(filled)])
    ccnt = np.concatenate([[0.0], np.cumsum(valid)])
    idx = np.arange(len(x))
    lo = np.maximum(0, idx + 1 - win)
    total = csum[idx + 1] - csum[lo]
    count = ccnt[idx + 1] - ccnt[lo]
    out = np.full(len(x), np.nan)
    ok = count >= max(1.0, 0.2 * win)
    out[ok] = total[ok] / count[ok]
    return out


def _mid_shoulder(kp):
    return (kp[:, L_SHOULDER] + kp[:, R_SHOULDER]) / 2.0


def _mid_hip(kp):
    return (kp[:, L_HIP] + kp[:, R_HIP]) / 2.0


def hand_speed(track, torso, fps, joint):
    rel = (track.kp[:, joint] - _mid_shoulder(track.kp)) / torso
    step = np.gradient(rel, 1.0 / fps, axis=0)
    return np.linalg.norm(step, axis=1)


def elbow_angle(kp, shoulder, elbow, wrist):
    a = kp[:, shoulder] - kp[:, elbow]
    b = kp[:, wrist] - kp[:, elbow]
    dot = np.sum(a * b, axis=1)
    mag = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9
    return np.degrees(np.arccos(np.clip(dot / mag, -1.0, 1.0)))


def lead_rear(values_left, values_right, lead_left):
    lead = np.where(lead_left, values_left, values_right)
    rear = np.where(lead_left, values_right, values_left)
    return lead, rear


def guard_up(kp):
    shoulder_y = _mid_shoulder(kp)[:, 1]
    up = (kp[:, L_WRIST, 1] < shoulder_y) & (kp[:, R_WRIST, 1] < shoulder_y)
    bad = np.isnan(kp[:, L_WRIST, 1]) | np.isnan(kp[:, R_WRIST, 1]) | np.isnan(shoulder_y)
    out = up.astype(float)
    out[bad] = np.nan
    return out


def pressure(kp_me, kp_opp, torso, fps):
    me = (_mid_shoulder(kp_me) + _mid_hip(kp_me)) / 2.0
    opp = (_mid_shoulder(kp_opp) + _mid_hip(kp_opp)) / 2.0
    toward = opp - me
    toward = toward / np.maximum(np.linalg.norm(toward, axis=1, keepdims=True), 1e-6)
    vel = np.gradient(me / torso, 1.0 / fps, axis=0)
    return np.sum(vel * toward, axis=1)


def fighter_distance(kp_a, kp_b, torso_a, torso_b):
    a = (_mid_shoulder(kp_a) + _mid_hip(kp_a)) / 2.0
    b = (_mid_shoulder(kp_b) + _mid_hip(kp_b)) / 2.0
    return np.linalg.norm(a - b, axis=1) / ((torso_a + torso_b) / 2.0)


def _sample(x, idx, ndigits):
    out = []
    for value in x[idx]:
        out.append(None if not np.isfinite(value) else round(float(value), ndigits))
    return out


def event_series(punches, fighter, times, t0, dt):
    mine = sorted((p for p in punches if p["fighter"] == fighter), key=lambda p: p["t"])
    ts = np.array([p["t"] for p in mine])
    landed = np.array([p["landed"] for p in mine], dtype=bool)
    speeds = np.array([p["speed"] for p in mine])
    out, land, speed = [], [], []
    for t in times:
        if len(ts) == 0:
            out.append(0.0)
            land.append(None)
            speed.append(None)
            continue
        window = max(MIN_OUTPUT_WINDOW_S, min(WINDOWS["output_s"], t - t0))
        in_out = (ts > t - WINDOWS["output_s"]) & (ts <= t)
        out.append(round(float(in_out.sum()) / window * 60.0, 1))
        in_land = (ts > t - WINDOWS["land_s"]) & (ts <= t)
        thrown = int(in_land.sum())
        land.append(
            round(float(landed[in_land].sum()) / thrown, 3) if thrown >= MIN_LAND_SAMPLE else None
        )
        upto = np.where(ts <= t)[0]
        recent = upto[-WINDOWS["speed_n"] :]
        speed.append(round(float(speeds[recent].mean()), 2) if len(recent) else None)
    return out, land, speed


def build_series(analysis, punches, fps, start_s, dt=SERIES_DT):
    tracks = analysis["tracks"]
    torsos = analysis["torsos"]
    leads = analysis["leads"]
    if tracks is None:
        return None
    total = tracks[0].kp.shape[0]
    n = int(np.floor((total - 1) / fps / dt)) + 1
    idx = np.minimum(total - 1, np.round(np.arange(n) * dt * fps).astype(int))
    times = start_s + np.arange(n) * dt
    series = {"dt": dt, "t0": round(float(start_s), 3), "n": int(n)}
    dist = trailing_mean(
        fighter_distance(tracks[0].kp, tracks[1].kp, torsos[0], torsos[1]),
        WINDOWS["dist_s"] * fps,
    )
    series["dist"] = _sample(dist, idx, 2)
    for i, fid in enumerate(FIGHTER_IDS):
        me, opp = tracks[i], tracks[1 - i]
        left_speed = hand_speed(me, torsos[i], fps, L_WRIST)
        right_speed = hand_speed(me, torsos[i], fps, R_WRIST)
        lead_speed, rear_speed = lead_rear(left_speed, right_speed, leads[i])
        left_angle = elbow_angle(me.kp, L_SHOULDER, L_ELBOW, L_WRIST)
        right_angle = elbow_angle(me.kp, R_SHOULDER, R_ELBOW, R_WRIST)
        lead_angle, _ = lead_rear(left_angle, right_angle, leads[i])
        lead_act = trailing_mean(lead_speed, WINDOWS["lead_act_s"] * fps)
        rear_act = trailing_mean(rear_speed, WINDOWS["lead_act_s"] * fps)
        angle = trailing_mean(lead_angle, WINDOWS["angle_s"] * fps)
        guard = trailing_mean(guard_up(me.kp), WINDOWS["guard_s"] * fps)
        press = trailing_mean(pressure(me.kp, opp.kp, torsos[i], fps), WINDOWS["pressure_s"] * fps)
        out, land, speed = event_series(punches, fid, times, start_s, dt)
        series[fid] = {
            "out": out,
            "land": land,
            "speed": speed,
            "lead_act": _sample(lead_act, idx, 2),
            "rear_act": _sample(rear_act, idx, 2),
            "lead_angle": _sample(angle, idx, 0),
            "guard": _sample(guard, idx, 3),
            "press": _sample(press, idx, 2),
            "lead_left": [bool(v) for v in leads[i][idx]],
        }
    return series


def build_pose(analysis, fps, start_s, width, height, hz=POSE_HZ):
    tracks = analysis["tracks"]
    if tracks is None:
        return None
    total = tracks[0].kp.shape[0]
    n = int(np.floor((total - 1) / fps * hz)) + 1
    idx = np.minimum(total - 1, np.round(np.arange(n) / hz * fps).astype(int))
    pose = {
        "hz": hz,
        "t0": round(float(start_s), 3),
        "w": int(width),
        "h": int(height),
        "joints": POSE_JOINTS,
    }
    for i, fid in enumerate(FIGHTER_IDS):
        rows = []
        for k in idx:
            pts = tracks[i].kp[k, POSE_JOINTS]
            if np.isnan(pts).all():
                rows.append(None)
                continue
            flat = []
            for x, y in pts:
                if np.isnan(x) or np.isnan(y):
                    flat.extend([-1, -1])
                else:
                    flat.append(int(round(x / width * 1000)))
                    flat.append(int(round(y / height * 1000)))
            rows.append(flat)
        pose[fid] = rows
    return pose
