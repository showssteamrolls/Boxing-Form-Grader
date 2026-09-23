"""
Regression tests for the pure math in features.py. These use synthetic
landmark arrays (no MediaPipe, no camera) so they run fast and stay stable
even if smoothing/thresholds get retuned.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pose_extraction import LANDMARK  # noqa: E402
from features import (  # noqa: E402
    _angle_deg,
    _elbow_extension,
    _hip_rotation,
    _guard_height_ratio,
    FeatureTracker,
)


def make_landmarks(**overrides) -> np.ndarray:
    """33x2 array, all zeros by default, with named landmarks overridden."""
    lm = np.zeros((33, 2), dtype=np.float32)
    for name, xy in overrides.items():
        lm[LANDMARK[name]] = xy
    return lm


def test_angle_deg_straight_line_is_180():
    a, b, c = np.array([0, 0]), np.array([1, 0]), np.array([2, 0])
    assert _angle_deg(a, b, c) == pytest.approx(180.0, abs=1e-2)


def test_angle_deg_right_angle_is_90():
    a, b, c = np.array([0, 1]), np.array([0, 0]), np.array([1, 0])
    assert _angle_deg(a, b, c) == pytest.approx(90.0, abs=1e-3)


def test_elbow_extension_fully_extended():
    lm = make_landmarks(
        LEFT_SHOULDER=(0, 0), LEFT_ELBOW=(1, 0), LEFT_WRIST=(2, 0),
    )
    assert _elbow_extension(lm, "left") == pytest.approx(180.0, abs=1e-3)


def test_elbow_extension_bent_90():
    lm = make_landmarks(
        LEFT_SHOULDER=(0, 1), LEFT_ELBOW=(0, 0), LEFT_WRIST=(1, 0),
    )
    assert _elbow_extension(lm, "left") == pytest.approx(90.0, abs=1e-3)


def test_hip_rotation_zero_when_hips_and_shoulders_parallel():
    lm = make_landmarks(
        LEFT_HIP=(0, 0), RIGHT_HIP=(1, 0),
        LEFT_SHOULDER=(0, -1), RIGHT_SHOULDER=(1, -1),
    )
    assert _hip_rotation(lm) == pytest.approx(0.0, abs=1e-3)


def test_hip_rotation_90_when_perpendicular():
    lm = make_landmarks(
        LEFT_HIP=(0, 0), RIGHT_HIP=(1, 0),           # horizontal hip line
        LEFT_SHOULDER=(0, 0), RIGHT_SHOULDER=(0, 1),  # vertical shoulder line
    )
    assert _hip_rotation(lm) == pytest.approx(90.0, abs=1e-3)


def test_guard_height_ratio_wrist_at_shoulder_level():
    lm = make_landmarks(
        LEFT_SHOULDER=(0, 0.5), LEFT_WRIST=(0, 0.5), LEFT_HIP=(0, 1.0),
    )
    assert _guard_height_ratio(lm, "left") == pytest.approx(1.0, abs=1e-3)


def test_guard_height_ratio_wrist_dropped_below_shoulder():
    # torso_len = 0.5; wrist is 0.25 *below* shoulder (larger y) -> ratio < 1
    lm = make_landmarks(
        LEFT_SHOULDER=(0, 0.5), LEFT_WRIST=(0, 0.75), LEFT_HIP=(0, 1.0),
    )
    assert _guard_height_ratio(lm, "left") == pytest.approx(0.5, abs=1e-3)


def test_feature_tracker_zero_velocity_on_first_frame():
    tracker = FeatureTracker()
    lm = make_landmarks(
        LEFT_SHOULDER=(0, 0), LEFT_WRIST=(1, 0),
        RIGHT_SHOULDER=(0, 0), RIGHT_WRIST=(1, 0),
        LEFT_HIP=(0, 1), RIGHT_HIP=(1, 1), LEFT_ELBOW=(0.5, 0), RIGHT_ELBOW=(0.5, 0),
    )
    feats = tracker.update(lm, timestamp_ms=0.0)
    assert feats.wrist_angular_velocity_dps["left"] == 0.0
    assert feats.wrist_angular_velocity_dps["right"] == 0.0


def test_feature_tracker_nonzero_velocity_after_rotation():
    tracker = FeatureTracker()
    base = dict(
        LEFT_SHOULDER=(0, 0), RIGHT_SHOULDER=(0, 0),
        LEFT_HIP=(0, 1), RIGHT_HIP=(1, 1),
        LEFT_ELBOW=(0.5, 0), RIGHT_ELBOW=(0.5, 0),
    )
    lm1 = make_landmarks(LEFT_WRIST=(1, 0), RIGHT_WRIST=(1, 0), **base)
    tracker.update(lm1, timestamp_ms=0.0)

    # Rotate the wrist 90 degrees over 100ms -> should register ~900 deg/s
    lm2 = make_landmarks(LEFT_WRIST=(0, 1), RIGHT_WRIST=(0, 1), **base)
    feats2 = tracker.update(lm2, timestamp_ms=100.0)

    assert feats2.wrist_angular_velocity_dps["left"] == pytest.approx(900.0, rel=0.05)
    assert feats2.wrist_angular_velocity_dps["right"] == pytest.approx(900.0, rel=0.05)
