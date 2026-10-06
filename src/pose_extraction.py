"""
Thin wrapper around MediaPipe's PoseLandmarker (Tasks API).

Responsibility boundary: this module ONLY turns a BGR frame into landmark
coordinates + visibility. It knows nothing about angles, smoothing, or
scoring — that separation is what lets thresholds/features get retuned
without touching this file.

Note on API version: MediaPipe removed the old `mp.solutions.pose` Python API
from recent releases (0.10.9+) in favor of the Tasks API used here
(`mediapipe.tasks.python.vision.PoseLandmarker`). If you're following an
older tutorial that uses `mp.solutions.pose.Pose(...)`, that API no longer
exists in current installs — this module is the up-to-date replacement.

Note on GPU delegate: forced to CPU below as a minor optimization, but this
does NOT avoid the real macOS crash risk. mediapipe 1.0.x has a known,
unresolved bug (google-ai-edge/mediapipe#6356) where TensorsToDetectionsCalculator
unconditionally constructs a Metal helper on macOS at the C++ level,
regardless of this Python-level delegate setting — it aborts the whole
process before any inference runs. The actual fix is the mediapipe version
pin in requirements.txt (<1.0): 0.10.x doesn't have this bug.
"""
from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass

import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

# Landmark indices (same 33-point topology MediaPipe has used since the
# original solutions API — indices are unchanged by the API migration).
# Full list: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
LANDMARK = {
    "LEFT_SHOULDER": 11, "RIGHT_SHOULDER": 12,
    "LEFT_ELBOW": 13, "RIGHT_ELBOW": 14,
    "LEFT_WRIST": 15, "RIGHT_WRIST": 16,
    "LEFT_HIP": 23, "RIGHT_HIP": 24,
    "LEFT_KNEE": 25, "RIGHT_KNEE": 26,
    "NOSE": 0,
}

# (start, end) index pairs for drawing the skeleton.
POSE_CONNECTIONS = [(c.start, c.end) for c in vision.PoseLandmarksConnections.POSE_LANDMARKS]

_MODEL_URLS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}


def ensure_model(model_path: str, variant: str = "lite") -> str:
    """Downloads the PoseLandmarker model bundle to `model_path` if it isn't
    already there. One-time ~5-30MB download depending on variant.
    Requires internet access on first run; after that it's cached on disk."""
    if os.path.exists(model_path):
        return model_path

    url = _MODEL_URLS[variant]
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    print(f"[pose_extraction] downloading {variant} pose model to {model_path} ...")
    try:
        urllib.request.urlretrieve(url, model_path)
    except Exception as e:
        raise RuntimeError(
            f"Could not download pose model from {url} ({e}). "
            f"If you're offline or the URL changed, download it manually and "
            f"place it at {model_path}. See: "
            f"https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker#models"
        ) from e
    return model_path


@dataclass
class PoseFrame:
    """One frame's worth of pose data, already in a plain-numpy form so
    downstream code never has to touch mediapipe's dataclass types."""
    timestamp_ms: float
    landmarks_xy: np.ndarray      # shape (33, 2), normalized image coords [0,1]
    visibility: np.ndarray        # shape (33,), 0..1 confidence per landmark
    detected: bool                # False if no pose found this frame


class PoseExtractor:
    def __init__(self, cfg: dict):
        pose_cfg = cfg["pose"]
        model_path = ensure_model(pose_cfg["model_path"], pose_cfg["model_variant"])

        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path, delegate=BaseOptions.Delegate.CPU),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=pose_cfg["min_detection_confidence"],
            min_tracking_confidence=pose_cfg["min_tracking_confidence"],
            min_pose_presence_confidence=pose_cfg["min_detection_confidence"],
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._last_timestamp_ms = -1  # VIDEO mode requires strictly increasing timestamps

    def process(self, frame_bgr: np.ndarray, timestamp_ms: float) -> PoseFrame:
        """Run pose detection on a single BGR frame (as read by cv2.VideoCapture)."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # VIDEO mode requires strictly increasing integer timestamps.
        ts_int = max(int(timestamp_ms), self._last_timestamp_ms + 1)
        self._last_timestamp_ms = ts_int

        result = self._landmarker.detect_for_video(mp_image, ts_int)

        if not result.pose_landmarks:
            return PoseFrame(
                timestamp_ms=timestamp_ms,
                landmarks_xy=np.zeros((33, 2), dtype=np.float32),
                visibility=np.zeros((33,), dtype=np.float32),
                detected=False,
            )

        lms = result.pose_landmarks[0]  # first (only, since num_poses=1) detected person
        xy = np.array([[lm.x, lm.y] for lm in lms], dtype=np.float32)
        # visibility can be None for some model variants; default to 1.0 (trust it)
        # rather than 0.0 (which would make smoothing.py permanently hold/freeze).
        vis = np.array([lm.visibility if lm.visibility is not None else 1.0 for lm in lms], dtype=np.float32)
        return PoseFrame(timestamp_ms=timestamp_ms, landmarks_xy=xy, visibility=vis, detected=True)

    def draw_landmarks(self, frame_bgr: np.ndarray, pose_frame: PoseFrame) -> None:
        """Mutates frame_bgr in place, drawing the skeleton overlay."""
        if not pose_frame.detected:
            return
        h, w = frame_bgr.shape[:2]
        pts = pose_frame.landmarks_xy
        vis = pose_frame.visibility
        for a, b in POSE_CONNECTIONS:
            if vis[a] < 0.3 or vis[b] < 0.3:
                continue
            pa = (int(pts[a][0] * w), int(pts[a][1] * h))
            pb = (int(pts[b][0] * w), int(pts[b][1] * h))
            cv2.line(frame_bgr, pa, pb, (235, 206, 135), 2)  # sky blue (BGR)
        for i, (x, y) in enumerate(pts):
            if vis[i] < 0.3:
                continue
            cv2.circle(frame_bgr, (int(x * w), int(y * h)), 3, (255, 0, 255), -1)  # magenta (BGR)

    def close(self) -> None:
        self._landmarker.close()
