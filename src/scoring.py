"""
Turns FrameFeatures into a 0-100 composite score, plus per-metric sub-scores
so the UI/overlay can show *why* a punch scored the way it did.

Grading model per metric:
- Inside [min_good, max_good]      -> sub-score 100
- Within `tolerance` past an edge  -> linear falloff to 0
- Beyond that                       -> sub-score 0
"""
from __future__ import annotations

from dataclasses import dataclass

from features import FrameFeatures


@dataclass
class PunchScore:
    composite: float                   # 0-100
    sub_scores: dict[str, float]       # per-metric, 0-100
    side: str                          # "left" or "right" — whichever arm was scored


def _metric_subscore(value: float, min_good: float, max_good: float, tolerance: float) -> float:
    if min_good <= value <= max_good:
        return 100.0
    if value < min_good:
        deficit = min_good - value
    else:
        deficit = value - max_good
    if deficit >= tolerance:
        return 0.0
    return 100.0 * (1.0 - deficit / tolerance)


def score_punch(features: FrameFeatures, cfg: dict, side: str) -> PunchScore:
    """Score a single arm's punch on this frame. `side` is 'left' or 'right' —
    callers decide which arm to score (typically whichever triggered punch
    detection in live_pipeline.py)."""
    thresholds = cfg["thresholds"]
    weights = cfg["score_weights"]

    metric_values = {
        "elbow_extension_deg": features.elbow_extension_deg[side],
        "hip_rotation_deg": features.hip_rotation_deg,
        "guard_height_ratio": features.guard_height_ratio["left" if side == "right" else "right"],
        "punch_angular_velocity_dps": features.wrist_angular_velocity_dps[side],
    }

    sub_scores = {}
    for metric, value in metric_values.items():
        t = thresholds[metric]
        sub_scores[metric] = _metric_subscore(value, t["min_good"], t["max_good"], t["tolerance"])

    composite = sum(sub_scores[m] * weights[m] for m in weights)
    return PunchScore(composite=composite, sub_scores=sub_scores, side=side)


class PunchDetector:
    """Flags when a punch is 'in progress' based on wrist angular velocity
    crossing a trigger threshold, with debouncing so one punch isn't counted
    multiple times across consecutive fast frames."""

    def __init__(self, cfg: dict):
        self.trigger_dps = cfg["punch_detection"]["wrist_velocity_trigger_dps"]
        self.min_frames_between = cfg["punch_detection"]["min_frames_between_punches"]
        self._frames_since_last: dict[str, int] = {"left": 999, "right": 999}

    def check(self, features: FrameFeatures) -> list[str]:
        """Returns list of sides ('left'/'right') that just triggered a new
        punch this frame."""
        triggered = []
        for side in ("left", "right"):
            self._frames_since_last[side] += 1
            speed = features.wrist_angular_velocity_dps[side]
            if speed >= self.trigger_dps and self._frames_since_last[side] >= self.min_frames_between:
                triggered.append(side)
                self._frames_since_last[side] = 0
        return triggered
