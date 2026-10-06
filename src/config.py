"""
Central config for the boxing form grader.

Everything a coach/user would want to retune lives here — nowhere else in the
codebase should hardcode a threshold, weight, or landmark index. Values can be
overridden by loading a YAML file with the same keys (see `load_config`).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    # --- MediaPipe Pose model settings ---
    "pose": {
        # MediaPipe's PoseLandmarker model bundle. "lite" is fastest and good
        # enough for real-time grading; use "full" or "heavy" if you have
        # GPU/CPU headroom and want more accurate landmarks on fast strikes.
        "model_variant": "lite",         # lite | full | heavy
        "model_path": "models/pose_landmarker_lite.task",  # auto-downloaded on first run
        "min_detection_confidence": 0.5,
        "min_tracking_confidence": 0.5,
    },

    # --- Visibility gating (occlusion handling) ---
    # A landmark below this visibility score is treated as "occluded" for that
    # frame: smoothing.py holds the last trusted value instead of updating.
    "min_landmark_visibility": 0.6,

    # --- EMA smoothing ---
    "smoothing": {
        "alpha": 0.35,          # weight on new sample; higher = less smoothing/lag
        "max_hold_frames": 8,   # give up holding and snap to raw after this many
                                  # consecutive occluded frames
    },

    # --- Feature thresholds (used by scoring.py) ---
    # Each metric maps to a (min_good, max_good) range. Values outside this
    # range reduce that metric's sub-score linearly up to `tolerance` degrees
    # or deg/s past the edge, then clamp to 0.
    # Calibrated from 90 self-labeled reps (20 cross, 70 jab) via
    # app/log_metrics.py + app/calibrate_thresholds.py, 10th/90th percentile.
    "thresholds": {
        "elbow_extension_deg": {"min_good": 95.8, "max_good": 129.2, "tolerance": 16.7},
        "hip_rotation_deg": {"min_good": 0.8, "max_good": 8.6, "tolerance": 3.9},
        "guard_height_ratio": {"min_good": 0.88, "max_good": 1.15, "tolerance": 0.13},
        "punch_angular_velocity_dps": {"min_good": 47.4, "max_good": 152.8, "tolerance": 52.7},
    },

    # --- Composite score weights (must sum to 1.0; validated on load) ---
    "score_weights": {
        "elbow_extension_deg": 0.35,
        "hip_rotation_deg": 0.30,
        "guard_height_ratio": 0.20,
        "punch_angular_velocity_dps": 0.15,
    },

    # --- Punch detection ---
    "punch_detection": {
        "elbow_extension_trigger_deg": 95,   # rising-edge extension = punch landed
        "elbow_extension_rearm_deg": 70,      # must drop back below this before the next punch can trigger
        "min_frames_between_punches": 6,      # debounce
    },

    # --- Display / overlay ---
    "overlay": {
        "draw_skeleton": True,
        "score_font_scale": 1.2,
        "good_color_bgr": [60, 200, 60],
        "bad_color_bgr": [40, 40, 220],
        "score_good_threshold": 75,
    },
}


def load_config(path: str | None = None) -> dict[str, Any]:
    """Return the default config, deep-merged with overrides from a YAML file.

    Unknown keys in the YAML are ignored with a warning rather than silently
    accepted, to catch typos in threshold names early.
    """
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if not path:
        _validate(cfg)
        return cfg

    with open(path, "r") as f:
        overrides = yaml.safe_load(f) or {}

    _deep_merge(cfg, overrides, prefix="")
    _validate(cfg)
    return cfg


def _deep_merge(base: dict, override: dict, prefix: str) -> None:
    for key, value in override.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if key not in base:
            print(f"[config] warning: unknown key '{full_key}' in override, ignoring")
            continue
        if isinstance(value, dict) and isinstance(base[key], dict):
            _deep_merge(base[key], value, full_key)
        else:
            base[key] = value


def _validate(cfg: dict[str, Any]) -> None:
    weight_sum = sum(cfg["score_weights"].values())
    if abs(weight_sum - 1.0) > 1e-6:
        raise ValueError(f"score_weights must sum to 1.0, got {weight_sum:.4f}")

    for metric in cfg["score_weights"]:
        if metric not in cfg["thresholds"]:
            raise ValueError(f"score_weights references unknown metric '{metric}'")
