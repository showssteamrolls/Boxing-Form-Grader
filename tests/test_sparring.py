import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sparring import (  # noqa: E402
    Detection,
    FighterTracker,
    SparringConfig,
    _landed,
    zone_series,
    _resolve_mutual_landings,
    head_center,
    head_ellipse,
    resolve_identity,
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
    conf[[1, 2, 3, 4, 7, 8]] = 0  # the synthetic fighters have no eyes, ears or elbows
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


def aimed_at_head(total, punches):
    a = make_fighter(total, 300, 1, punches)
    a[:, L_WR, 1] = 172  # jab travels at head height
    return a


def test_landed_depends_on_distance_to_opponent():
    total = 80
    far = analyze(to_pairs(aimed_at_head(total, [(20, "left", "straight")]), make_fighter(total, 700, -1)), FPS)[0]
    near = analyze(to_pairs(aimed_at_head(total, [(20, "left", "straight")]), make_fighter(total, 420, -1)), FPS)[0]
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
    tracker = FighterTracker(FPS, SparringConfig(color_weight=1.5))
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


def test_long_detection_gap_does_not_blank_later_punches():
    total = 400
    a = make_fighter(total, 300, 1, [(300, "left", "straight"), (340, "left", "straight")])
    b = make_fighter(total, 700, -1)
    punches, _ = analyze(to_pairs(a, b, drop_a=set(range(50, 120))), FPS)
    mine = [p for p in punches if p["fighter"] == "A"]
    assert len(mine) == 2
    assert all(p["t"] > 9.0 for p in mine)


def test_low_confidence_wrist_gap_does_not_blank_later_frames():
    total = 300
    a = make_fighter(total, 300, 1, [(250, "left", "straight")])
    b = make_fighter(total, 700, -1)
    pairs = to_pairs(a, b)
    for t in range(40, 90):
        pairs[t][0].conf[L_WR] = 0.05
    punches, _ = analyze(pairs, FPS)
    assert len([p for p in punches if p["fighter"] == "A"]) == 1


def test_brightness_identity_survives_crossing():
    tracker = FighterTracker(FPS, SparringConfig(lum_weight=6.0), lum_refs=(0.05, 0.2))
    steps = 120
    dark_side = []
    for t in range(steps + 1):
        frac = t / steps
        dark = box_det(200 + 600 * frac, color=(0.05,) * 3 + (0.3,) * 3)
        light = box_det(800 - 600 * frac, color=(0.2,) * 3 + (0.3,) * 3)
        out = tracker.update([light, dark])
        dark_side.append(out[0].color[0] < 0.1 if out[0] is not None else None)
    assert all(d for d in dark_side if d is not None)


def test_resolve_identity_puts_darker_shirt_in_slot_zero():
    def det(cx, shade):
        d = box_det(cx, color=(shade,) * 3 + (0.3,) * 3)
        return d
    pairs = []
    for t in range(40):
        pairs.append([det(200, 0.2), det(800, 0.05)])  # slot 0 is the light shirt: wrong way round
    fixed, changes = resolve_identity(pairs)
    assert fixed[0][0].color[0] < 0.1
    assert fixed[0][1].color[0] > 0.1
    assert changes == 0


def opponent(elbow_l=(485, 250), elbow_r=(515, 250), wrist_l=(480, 215), wrist_r=(520, 215)):
    """Single-frame opponent facing left: nose (480,170), shoulders y=200, hips y=300, torso length 100."""
    kp = np.zeros((1, 17, 2))
    kp[0, 0] = (480, 170)
    kp[0, L_SH] = (480, 200)
    kp[0, R_SH] = (520, 200)
    kp[0, L_HIP] = (480, 300)
    kp[0, R_HIP] = (520, 300)
    kp[0, 7], kp[0, 8] = elbow_l, elbow_r
    kp[0, L_WR], kp[0, R_WR] = wrist_l, wrist_r
    return type("Opp", (), {"kp": kp})()


def hits(wrist, **kw):
    opp = opponent(**kw)
    zones = zone_series(opp, 100.0, opp)
    return _landed(opp, 100.0, np.array([wrist], dtype=float), 0, 0, SparringConfig(), zones=zones)


def broad_opponent(front=True):
    """Opponent with shoulders and hips 80 px wide, chest to the camera (front) or back to it."""
    opp = opponent(wrist_l=(440, 140), wrist_r=(560, 140), elbow_l=(430, 250), elbow_r=(570, 250))
    left, right = (540, 460) if front else (460, 540)
    opp.kp[0, L_SH], opp.kp[0, R_SH] = (left, 200), (right, 200)
    opp.kp[0, L_HIP], opp.kp[0, R_HIP] = (left, 300), (right, 300)
    opp.kp[0, 0] = (500, 170)
    if front:
        opp.kp[0, 1:3] = [(492, 165), (508, 165)]
    else:
        opp.kp[0, 0:3] = np.nan
        opp.kp[0, 3:5] = [(475, 170), (525, 170)]
    return opp


def broad_hit(opp, wrist):
    zones = zone_series(opp, 100.0, opp)
    return _landed(opp, 100.0, np.array([wrist], dtype=float), 0, 0, SparringConfig(), zones=zones)


def test_landed_counts_head_and_body_zones():
    assert hits((482, 172))  # on the face
    front = broad_opponent()
    assert broad_hit(front, (500, 232)) == "solar_plexus"
    assert broad_hit(front, (500, 272)) == "core"


def test_landed_ignores_chest_and_back():
    assert broad_hit(broad_opponent(), (500, 202)) is None  # chest line, not a target
    back = broad_opponent(front=False)
    assert broad_hit(back, (500, 232)) is None
    assert broad_hit(back, (500, 272)) is None


def test_zones_follow_a_body_turned_to_the_side():
    front = broad_opponent()
    turned = broad_opponent()
    turned.kp[0, L_SH], turned.kp[0, R_SH] = (505, 200), (495, 200)  # shoulders nearly stacked
    turned.kp[0, L_HIP], turned.kp[0, R_HIP] = (505, 300), (495, 300)
    zf = zone_series(front, 100.0, front)[0]
    zt = zone_series(turned, 100.0, turned)[0]
    assert zt[1][2] <= zf[1][2]  # liver is no wider when seen from the side
    assert np.all(np.isfinite(zt))  # still drawn, not dropped


def test_landed_ignores_shoulder_hits():
    assert not hits((482, 202), wrist_l=(470, 140), wrist_r=(530, 140))


def test_landed_ignores_arm_and_block_hits():
    # a glove on the forearm in front of the body is a block, not a landed shot
    assert not hits((500, 215), wrist_l=(470, 140), wrist_r=(530, 140), elbow_l=(500, 260), elbow_r=(520, 260))


def test_landed_ignores_contact_beside_the_torso():
    assert not hits((590, 250), wrist_l=(470, 140), wrist_r=(530, 140))


def test_head_target_holds_when_the_nose_drops_out():
    opp = opponent()
    kp = opp.kp[0].copy()
    with_nose = head_center(kp, 100.0).copy()
    kp[0] = np.nan
    without_nose = head_center(kp, 100.0)
    assert np.linalg.norm(with_nose - without_nose) < 30  # same place, within a third of a torso


def test_landed_still_counts_when_a_glove_hides_the_nose():
    opp = opponent(wrist_l=(470, 140), wrist_r=(530, 140))
    opp.kp[0, 0] = np.nan
    assert _landed(opp, 100.0, np.array([[500, 160.0]]), 0, 0, SparringConfig())


def test_mutual_landings_keep_only_the_faster_punch():
    punches = [
        {"t": 10.0, "fighter": "A", "landed": True, "speed": 11.8},
        {"t": 10.1, "fighter": "B", "landed": True, "speed": 6.0},
        {"t": 20.0, "fighter": "B", "landed": True, "speed": 5.0},
    ]
    _resolve_mutual_landings(punches, 0.3)
    assert [p["landed"] for p in punches] == [True, False, True]


def test_fighters_at_different_depths_do_not_land():
    total = 80
    near = analyze(to_pairs(aimed_at_head(total, [(20, "left", "straight")]), make_fighter(total, 420, -1)), FPS)[0]
    assert near[0]["landed"] is True
    b = make_fighter(total, 420, -1)
    b[:, :, 1] -= 80  # same image position but far deeper in the scene: boxes sit much higher
    a = aimed_at_head(total, [(20, "left", "straight")])
    a[:, L_WR, 1] = 92
    deep = analyze(to_pairs(a, b), FPS)[0]
    assert deep[0]["landed"] is False


def face_kp(nose_rise):
    """Upright torso (length 100, shoulders at y=200); the nose sits nose_rise torso lengths above the shoulders."""
    kp = np.full((17, 2), np.nan)
    kp[L_SH], kp[R_SH] = (480, 200), (520, 200)
    kp[L_HIP], kp[R_HIP] = (480, 300), (520, 300)
    ny = 200 - nose_rise * 100
    kp[0] = (500, ny)
    kp[1], kp[2] = (490, ny - 12), (510, ny - 12)
    kp[3], kp[4] = (475, ny - 8), (525, ny - 8)
    return kp


def test_face_target_shrinks_when_the_chin_tucks():
    up = head_ellipse(face_kp(0.55), 100.0, ref_rise=0.55)
    down = head_ellipse(face_kp(0.30), 100.0, ref_rise=0.55)
    assert down[3] < up[3] * 0.8  # shorter
    assert abs(down[2] - up[2]) < 12  # width follows the ears, which did not move


def test_face_target_covers_forehead_through_chin():
    cx, cy, rx, ry = head_ellipse(face_kp(0.55), 100.0, ref_rise=0.55)
    eyes_y = 145 - 12
    assert cy - ry < eyes_y - 15 and cy + ry > 145 + 10  # forehead above the eyes, chin below the nose


def test_face_target_ignores_a_glove_mistaken_for_an_ear():
    kp = face_kp(0.55)
    kp[3] = (300, 150)
    cx, cy, rx, ry = head_ellipse(kp, 100.0, ref_rise=0.55)
    assert abs(cx - 500) < 30


def test_landed_ignores_a_parry_on_the_glove_in_front_of_the_face():
    # glove lands inside the face zone but right on the opponent's raised glove
    assert not hits((484, 176), wrist_l=(488, 180), wrist_r=(530, 140))
    # the same spot with the guard dropped is a hit
    assert hits((484, 176), wrist_l=(470, 260), wrist_r=(530, 260), elbow_l=(470, 280), elbow_r=(530, 280))
