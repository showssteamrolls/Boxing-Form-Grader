"""
Turns smoothed landmark coordinates into the numeric features scoring.py
grades against: joint angles, angular velocity, and guard height.

All angle math works in normalized image coordinates (x, y in [0,1]) rather
than pixels — angles are scale-invariant so this is fine, but note that
because normalized x/y aren't equal-aspect unless the frame is square, angles
computed this way are an approximation. That's acceptable for form grading
(coach-level tolerance, not biomechanics-lab precision); flag it if you ever
need sub-degree accuracy.

Angular velocity requires two frames' worth of state, so `FeatureTracker` is
stateful — construct one per tracked person and call `update()` once per
frame with that frame's *timestamp*, not just its index, since webcam frame
intervals aren't perfectly uniform.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from pose_extraction import LANDMARK


@dataclass
class FrameFeatures:
    elbow_extension_deg: dict[str, float]      # {"left": ..., "right": ...}
    hip_rotation_deg: float
    guard_height_ratio: dict[str, float]       # {"left": ..., "right": ...}
    wrist_angular_velocity_dps: dict[str, float]  # {"left": ..., "right": ...}


def _angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at vertex b, formed by points a-b-c, in degrees."""
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-8
    cos_angle = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def _elbow_extension(landmarks_xy: np.ndarray, side: str) -> float:
    shoulder = landmarks_xy[LANDMARK[f"{side.upper()}_SHOULDER"]]
    elbow = landmarks_xy[LANDMARK[f"{side.upper()}_ELBOW"]]
    wrist = landmarks_xy[LANDMARK[f"{side.upper()}_WRIST"]]
    return _angle_deg(shoulder, elbow, wrist)


def _hip_rotation(landmarks_xy: np.ndarray) -> float:
    """Approximates hip rotation as the angle of the hip line relative to the
    shoulder line (in-plane proxy for torso twist — a true 3D rotation needs
    depth, which MediaPipe's 2D landmarks don't reliably give from one camera)."""
    l_hip, r_hip = landmarks_xy[LANDMARK["LEFT_HIP"]], landmarks_xy[LANDMARK["RIGHT_HIP"]]
    l_sh, r_sh = landmarks_xy[LANDMARK["LEFT_SHOULDER"]], landmarks_xy[LANDMARK["RIGHT_SHOULDER"]]

    hip_vec = r_hip - l_hip
    shoulder_vec = r_sh - l_sh
    hip_angle = np.degrees(np.arctan2(hip_vec[1], hip_vec[0]))
    shoulder_angle = np.degrees(np.arctan2(shoulder_vec[1], shoulder_vec[0]))
    diff = abs(hip_angle - shoulder_angle)
    return float(min(diff, 360 - diff))


def _guard_height_ratio(landmarks_xy: np.ndarray, side: str) -> float:
    """Wrist height relative to shoulder height, normalized by torso length
    (shoulder-to-hip), so it's roughly build/frame-size invariant.
    ~1.0 means wrist near shoulder height (a reasonable default guard);
    calibrate min_good/max_good in config for your own stance."""
    shoulder = landmarks_xy[LANDMARK[f"{side.upper()}_SHOULDER"]]
    wrist = landmarks_xy[LANDMARK[f"{side.upper()}_WRIST"]]
    hip = landmarks_xy[LANDMARK[f"{side.upper()}_HIP"]]

    torso_len = abs(hip[1] - shoulder[1]) + 1e-6
    # Image y grows downward, so a wrist above the shoulder has wrist[1] < shoulder[1].
    height_above_shoulder = shoulder[1] - wrist[1]
    return float(1.0 + height_above_shoulder / torso_len)


class FeatureTracker:
    def __init__(self):
        self._prev_wrist_xy: dict[str, Optional[np.ndarray]] = {"left": None, "right": None}
        self._prev_timestamp_ms: Optional[float] = None

    def reset(self) -> None:
        self._prev_wrist_xy = {"left": None, "right": None}
        self._prev_timestamp_ms = None

    def update(self, landmarks_xy: np.ndarray, timestamp_ms: float) -> FrameFeatures:
        elbow_extension = {
            side: _elbow_extension(landmarks_xy, side) for side in ("left", "right")
        }
        guard_height = {
            side: _guard_height_ratio(landmarks_xy, side) for side in ("left", "right")
        }
        hip_rotation = _hip_rotation(landmarks_xy)
        angular_velocity = self._wrist_angular_velocity(landmarks_xy, timestamp_ms)

        self._prev_timestamp_ms = timestamp_ms
        return FrameFeatures(
            elbow_extension_deg=elbow_extension,
            hip_rotation_deg=hip_rotation,
            guard_height_ratio=guard_height,
            wrist_angular_velocity_dps=angular_velocity,
        )

    def _wrist_angular_velocity(self, landmarks_xy: np.ndarray, timestamp_ms: float) -> dict[str, float]:
        """Degrees/second the shoulder->wrist vector swept between frames.
        This tracks *rotational* speed of the arm (useful for punch snap),
        not linear wrist speed in pixels/sec.

        Guarded against short shoulder->wrist vectors (e.g. wrist resting near
        the body in a guard position): a short vector's angle is unstable
        under small landmark jitter, which otherwise fakes huge angular
        velocity spikes with no real punch behind them. Only used for scoring
        now, not as the punch-detection trigger (see PunchDetector)."""
        MIN_VECTOR_NORM = 0.08  # normalized coords; below this the angle is unstable

        result = {}
        dt_s = None
        if self._prev_timestamp_ms is not None:
            dt_s = (timestamp_ms - self._prev_timestamp_ms) / 1000.0

        for side in ("left", "right"):
            shoulder = landmarks_xy[LANDMARK[f"{side.upper()}_SHOULDER"]]
            wrist = landmarks_xy[LANDMARK[f"{side.upper()}_WRIST"]]
            curr_vec = wrist - shoulder
            curr_norm = np.linalg.norm(curr_vec)

            prev = self._prev_wrist_xy[side]
            prev_norm = np.linalg.norm(prev) if prev is not None else 0.0

            if curr_norm < MIN_VECTOR_NORM or prev_norm < MIN_VECTOR_NORM or prev is None or not dt_s or dt_s <= 0:
                result[side] = 0.0
            else:
                curr_angle = np.degrees(np.arctan2(curr_vec[1], curr_vec[0]))
                prev_angle = np.degrees(np.arctan2(prev[1], prev[0]))
                delta = abs(curr_angle - prev_angle)
                delta = min(delta, 360 - delta)
                result[side] = float(delta / dt_s)

            self._prev_wrist_xy[side] = curr_vec

        return result
