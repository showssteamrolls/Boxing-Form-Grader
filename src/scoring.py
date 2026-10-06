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
    """Flags a punch on the rising edge of elbow extension (not velocity —
    velocity is noisiest exactly when a punch starts, since the shoulder-
    wrist vector is short near the guard).

    Uses hysteresis, not a single threshold: once triggered, a side can't
    trigger again until extension drops below `extension_rearm_deg` (well
    below the trigger). A single rise-then-retract threshold let landmark
    jitter during retraction bounce back above the trigger and double-count
    one real punch as two — confirmed on real footage (70 detected vs ~38
    actual reps, roughly the 2x pattern jitter-bounce produces)."""

    def __init__(self, cfg: dict):
        self.extension_trigger_deg = cfg["punch_detection"]["elbow_extension_trigger_deg"]
        self.extension_rearm_deg = cfg["punch_detection"]["elbow_extension_rearm_deg"]
        self.min_frames_between = cfg["punch_detection"]["min_frames_between_punches"]
        self._frames_since_last: dict[str, int] = {"left": 999, "right": 999}
        self._armed: dict[str, bool] = {"left": True, "right": True}

    def check(self, features: FrameFeatures) -> list[str]:
        """Returns list of sides ('left'/'right') that just triggered a new
        punch this frame."""
        triggered = []
        for side in ("left", "right"):
            self._frames_since_last[side] += 1
            extension = features.elbow_extension_deg[side]

            if not self._armed[side] and extension <= self.extension_rearm_deg:
                self._armed[side] = True

            if (
                self._armed[side]
                and extension >= self.extension_trigger_deg
                and self._frames_since_last[side] >= self.min_frames_between
            ):
                triggered.append(side)
                self._frames_since_last[side] = 0
                self._armed[side] = False
        return triggered
