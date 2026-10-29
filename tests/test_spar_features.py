import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from test_sparring import FPS, L_WR, R_WR, make_fighter, to_pairs  # noqa: E402
from sparring import analyze_full, lead_series, stack_tracks, clean_track, SparringConfig  # noqa: E402
from spar_features import (  # noqa: E402
    build_pose,
    build_series,
    elbow_angle,
    event_series,
    guard_up,
    trailing_mean,
)


def run(a, b, start_s=0.0):
    analysis = analyze_full(to_pairs(a, b), FPS, SparringConfig(), start_s)
    return analysis, build_series(analysis, analysis["punches"], FPS, start_s)


def test_trailing_mean_is_nan_aware_and_windowed():
    x = np.array([1.0, 2.0, 3.0, np.nan, 5.0, 6.0])
    out = trailing_mean(x, 3)
    assert out[2] == pytest.approx(2.0)
    assert out[3] == pytest.approx(2.5)
    assert out[5] == pytest.approx(5.5)


def test_elbow_angle_straight_and_right_angle():
    kp = np.zeros((2, 17, 2))
    kp[0, 5], kp[0, 7], kp[0, 9] = (0, 0), (1, 0), (2, 0)
    kp[1, 5], kp[1, 7], kp[1, 9] = (0, 0), (1, 0), (1, 1)
    angles = elbow_angle(kp, 5, 7, 9)
    assert angles[0] == pytest.approx(180.0, abs=0.01)
    assert angles[1] == pytest.approx(90.0, abs=0.01)


def test_guard_up_requires_both_wrists_above_shoulders():
    kp = np.zeros((3, 17, 2))
    kp[:, 5] = (0, 200)
    kp[:, 6] = (10, 200)
    kp[0, L_WR], kp[0, R_WR] = (0, 150), (10, 150)
    kp[1, L_WR], kp[1, R_WR] = (0, 150), (10, 260)
    kp[2, L_WR], kp[2, R_WR] = (np.nan, np.nan), (10, 150)
    out = guard_up(kp)
    assert out[0] == 1.0
    assert out[1] == 0.0
    assert np.isnan(out[2])


def test_event_series_rolling_output_land_and_speed():
    punches = [
        {"t": 1.0, "fighter": "A", "landed": True, "speed": 10.0},
        {"t": 2.0, "fighter": "A", "landed": False, "speed": 12.0},
        {"t": 3.0, "fighter": "A", "landed": True, "speed": 14.0},
        {"t": 4.0, "fighter": "B", "landed": True, "speed": 8.0},
    ]
    times = np.array([0.5, 3.5, 40.0])
    out, land, speed = event_series(punches, "A", times, 0.0, 0.25)
    assert out[0] == 0.0
    assert land[0] is None and speed[0] is None
    assert out[1] == pytest.approx(3 / 5.0 * 60.0, abs=0.1)
    assert land[1] == pytest.approx(2 / 3, abs=0.001)
    assert speed[1] == pytest.approx(12.0)
    assert out[2] == 0.0
    assert land[2] is None


def test_active_lead_hand_scores_higher_than_idle():
    total = 300
    jabs = [(20 + 25 * i, "left", "straight") for i in range(10)]
    a = make_fighter(total, 300, 1, jabs)
    b = make_fighter(total, 700, -1)
    _, series = run(a, b)
    a_act = np.nanmean([v for v in series["A"]["lead_act"] if v is not None])
    b_act = np.nanmean([v for v in series["B"]["lead_act"] if v is not None])
    assert a_act > b_act + 0.5
    assert b_act < 0.3


def test_series_shape_and_distance():
    total = 120
    a = make_fighter(total, 300, 1)
    b = make_fighter(total, 700, -1)
    analysis, series = run(a, b)
    assert series["n"] == len(series["dist"]) == len(series["A"]["out"]) == len(series["B"]["guard"])
    assert series["dt"] == 0.25
    dist = [v for v in series["dist"] if v is not None]
    assert dist and all(abs(d - dist[0]) < 0.05 for d in dist)
    assert dist[0] == pytest.approx(400 / 100, rel=0.05)


def test_pressure_is_positive_when_advancing():
    total = 150
    a = make_fighter(total, 300, 1)
    b = make_fighter(total, 700, -1)
    shift = np.linspace(0, 200, total)
    a[:, :, 0] += shift[:, None]
    _, series = run(a, b)
    a_press = [v for v in series["A"]["press"] if v is not None]
    b_press = [v for v in series["B"]["press"] if v is not None]
    assert np.mean(a_press) == pytest.approx(0.4, abs=0.05)
    assert abs(np.mean(b_press)) < 0.05


def test_lead_hand_follows_a_stance_switch():
    total = 240
    a = make_fighter(total, 300, 1)
    a[120:, L_WR, 0], a[120:, R_WR, 0] = a[120:, R_WR, 0].copy(), a[120:, L_WR, 0].copy()
    b = make_fighter(total, 700, -1)
    tracks = [clean_track(t, FPS, SparringConfig()) for t in stack_tracks(to_pairs(a, b))]
    lead = lead_series(tracks[0], tracks[1], 100.0, FPS)
    assert lead[:90].all()
    assert not lead[150:].any()


def test_punch_flags_lead_hand_and_uses_lead_at_the_time():
    total = 200
    a = make_fighter(total, 300, 1, [(20, "left", "straight"), (60, "right", "straight")])
    b = make_fighter(total, 700, -1)
    analysis, _ = run(a, b)
    mine = [p for p in analysis["punches"] if p["fighter"] == "A"]
    assert [p["lead"] for p in mine] == [True, False]
    assert [p["type"] for p in mine] == ["jab", "power"]


def test_pose_export_scales_and_marks_missing():
    total = 60
    a = make_fighter(total, 300, 1)
    b = make_fighter(total, 700, -1)
    analysis = analyze_full(to_pairs(a, b, drop_a=set(range(0, 60))), FPS, SparringConfig(), 5.0)
    pose = build_pose(analysis, FPS, 5.0, 1000, 500)
    assert pose["t0"] == 5.0 and pose["hz"] == 15
    assert len(pose["A"]) == len(pose["B"])
    assert all(row is None for row in pose["A"])
    row = pose["B"][0]
    assert len(row) == 18
    assert row[0] == 700 + (-25) * 1 or abs(row[0] - 675) <= 1
