import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sparring import (  # noqa: E402
    Detection,
    FighterTracker,
    SparringConfig,
    analyze,
    build_report,
    summarize,
)

FPS = 30.0
NOSE, L_SH, R_SH, L_WR, R_WR, L_HIP, R_HIP = 0, 5, 6, 9, 10, 11, 12


def min_jerk(n):
    s = np.linspace(0, 1, n)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def punch_profile(out_frames=6, back_frames=7):
    out = min_jerk(out_frames)[1:]
    back = 1 - min_jerk(back_frames)[1:]
    return np.concatenate([out, [1.0], back])


def make_fighter(total, x, facing, punches=(), noise=0.0, seed=0, slow=False):
    rng = np.random.default_rng(seed)
    kp = np.zeros((total, 17, 2))
    kp[:, NOSE] = (x + 25 * facing, 170)
    kp[:, L_SH] = (x - 8, 200)
    kp[:, R_SH] = (x + 8, 200)
    kp[:, L_HIP] = (x - 8, 300)
    kp[:, R_HIP] = (x + 8, 300)
    kp[:, L_WR] = (x + 55 * facing, 190)
    kp[:, R_WR] = (x + 30 * facing, 190)
    for frame, hand, kind in punches:
        joint = L_WR if hand == "left" else R_WR
        if slow:
            s = np.linspace(0, 1, 40)
        else:
            s = punch_profile()
        for k, value in enumerate(s):
            t = frame + k
            if t >= total:
                break
            forward = (70 if kind == "hook" else 80) * value * facing
            lateral = -50 * np.sin(np.pi * min(value, 1.0)) if kind == "hook" else 0.0
            kp[t, joint, 0] += forward
            kp[t, joint, 1] += lateral
    if noise:
        kp += rng.normal(0, noise, kp.shape)
    return kp


def to_pairs(kp_a, kp_b, drop_a=(), drop_b=()):
    total = kp_a.shape[0]
    conf = np.ones(17)
    pairs = []
    for t in range(total):
        pair = []
        for kp, drop in ((kp_a, drop_a), (kp_b, drop_b)):
            if t in drop:
                pair.append(None)
                continue
            xs, ys = kp[t, :, 0], kp[t, :, 1]
            box = np.array([xs.min() - 30, ys.min() - 40, xs.max() + 30, ys.max() + 30])
            pair.append(Detection(kp=kp[t].copy(), conf=conf.copy(), box=box))
        pairs.append(pair)
    return pairs


def expected_t(frame):
    return (frame + 3) / FPS


def test_detects_and_attributes_each_fighter():
    total = 150
    a = make_fighter(total, 300, 1, [(20, "left", "straight"), (60, "left", "straight"), (100, "left", "straight")])
    b = make_fighter(total, 700, -1, [(40, "right", "straight"), (80, "right", "straight")])
    punches, stances = analyze(to_pairs(a, b), FPS)
    mine = [p for p in punches if p["fighter"] == "A"]
    theirs = [p for p in punches if p["fighter"] == "B"]
    assert len(mine) == 3
    assert len(theirs) == 2
    assert all(p["type"] == "jab" and p["hand"] == "left" for p in mine)
    assert all(p["type"] == "power" and p["hand"] == "right" for p in theirs)
    for p, frame in zip(mine, (20, 60, 100)):
        assert abs(p["t"] - expected_t(frame)) < 0.2
    assert stances == ["orthodox", "orthodox"]


def test_hook_with_lead_hand_counts_as_power():
    total = 80
    a = make_fighter(total, 300, 1, [(20, "left", "hook")])
    b = make_fighter(total, 700, -1)
    punches, _ = analyze(to_pairs(a, b), FPS)
    assert len(punches) == 1
    assert punches[0]["type"] == "power"


def test_idle_with_noise_throws_nothing():
    total = 240
    a = make_fighter(total, 300, 1, noise=1.0, seed=1)
    b = make_fighter(total, 700, -1, noise=1.0, seed=2)
    punches, _ = analyze(to_pairs(a, b), FPS)
    assert punches == []


def test_slow_arm_movement_is_ignored():
    total = 120
    a = make_fighter(total, 300, 1, [(20, "left", "straight")], slow=True)
    b = make_fighter(total, 700, -1)
    punches, _ = analyze(to_pairs(a, b), FPS)
    assert punches == []


def test_landed_depends_on_distance_to_opponent():
    total = 80
    far = analyze(
        to_pairs(
            make_fighter(total, 300, 1, [(20, "left", "straight")]),
            make_fighter(total, 700, -1),
        ),
        FPS,
    )[0]
    near = analyze(
        to_pairs(
            make_fighter(total, 300, 1, [(20, "left", "straight")]),
            make_fighter(total, 420, -1),
        ),
        FPS,
    )[0]
    assert len(far) == 1 and far[0]["landed"] is False
    assert len(near) == 1 and near[0]["landed"] is True


def test_short_detection_gap_before_a_punch_is_bridged():
    total = 100
    a = make_fighter(total, 300, 1, [(40, "left", "straight")])
    b = make_fighter(total, 700, -1)
    punches, _ = analyze(to_pairs(a, b, drop_a={30, 31}), FPS)
    assert len(punches) == 1


def test_southpaw_stance_is_detected():
    total = 60
    a = make_fighter(total, 300, 1)
    a[:, L_WR, 0], a[:, R_WR, 0] = a[:, R_WR, 0].copy(), a[:, L_WR, 0].copy()
    b = make_fighter(total, 700, -1)
    _, stances = analyze(to_pairs(a, b), FPS)
    assert stances[0] == "southpaw"
    assert stances[1] == "orthodox"


def test_start_offset_shifts_timestamps():
    total = 80
    a = make_fighter(total, 300, 1, [(20, "left", "straight")])
    b = make_fighter(total, 700, -1)
    base, _ = analyze(to_pairs(a, b), FPS)
    shifted, _ = analyze(to_pairs(a, b), FPS, start_s=120.0)
    assert shifted[0]["t"] == pytest.approx(base[0]["t"] + 120.0)


def box_det(cx, cy=230, color=None, w=120, h=240):
    box = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
    kp = np.zeros((17, 2))
    kp[L_SH] = (cx - 8, cy - 30)
    kp[R_SH] = (cx + 8, cy - 30)
    kp[L_HIP] = (cx - 8, cy + 70)
    kp[R_HIP] = (cx + 8, cy + 70)
    return Detection(kp=kp, conf=np.ones(17), box=box, color=None if color is None else np.array(color, dtype=float))


def test_tracker_keeps_identity_when_detection_order_flips():
    tracker = FighterTracker(FPS)
    first = tracker.update([box_det(200), box_det(800)])
    assert first[0].box[0] < first[1].box[0]
    second = tracker.update([box_det(805), box_det(205)])
    assert second[0].box[0] < 300
    assert second[1].box[0] > 600


def test_tracker_ignores_small_bystander():
    tracker = FighterTracker(FPS)
    tracker.update([box_det(200), box_det(800)])
    bystander = box_det(500, cy=260, w=40, h=80)
    out = tracker.update([box_det(205), bystander, box_det(795)])
    assert out[0].box[0] < 300
    assert out[1].box[0] > 600


def test_tracker_uses_color_when_fighters_cross():
    tracker = FighterTracker(FPS)
    red, blue = (0.1, 0.1, 0.9), (0.9, 0.1, 0.1)
    steps = 100
    last = None
    for t in range(steps + 1):
        frac = t / steps
        a = box_det(200 + 600 * frac, color=red)
        b = box_det(800 - 600 * frac, color=blue)
        last = tracker.update([b, a])
    assert last[0].color[2] > 0.5
    assert last[0].box[0] > 600
    assert last[1].color[0] > 0.5
    assert last[1].box[0] < 300


def test_tracker_returns_none_before_two_fighters_are_seen():
    tracker = FighterTracker(FPS)
    assert tracker.update([box_det(200)]) == [None, None]


def test_summary_and_report_shape():
    punches = [
        {"t": 1.0, "fighter": "A", "hand": "left", "type": "jab", "landed": True, "speed": 9.0},
        {"t": 2.0, "fighter": "A", "hand": "right", "type": "power", "landed": False, "speed": 12.0},
        {"t": 2.5, "fighter": "B", "hand": "right", "type": "power", "landed": True, "speed": 11.0},
    ]
    counts = summarize(punches)
    assert counts["A"] == {"thrown": 2, "landed": 1, "jab": 1, "power": 1}
    assert counts["B"] == {"thrown": 1, "landed": 1, "jab": 0, "power": 1}
    report = build_report(punches, ["orthodox", "southpaw"], ["Me", "Partner"], 30.0, 0, 60, {"title": "t"})
    assert report["fighters"][1] == {"id": "B", "label": "Partner", "stance": "southpaw"}
    assert report["counts"] == counts
