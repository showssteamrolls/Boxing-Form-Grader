"""
Exponential moving average smoothing for landmark coordinates, gated on
per-landmark visibility.

Why gated: MediaPipe still emits a coordinate for a landmark it can't really
see (e.g. a fist crossing in front of the torso) — it's just a low-confidence
guess. Smoothing that guess in blindly gives you a confidently wrong,
smoothed-looking trajectory. Instead, when visibility drops below threshold
we hold the last trusted (visible) value, and only resume updating once the
landmark is visible again — or after `max_hold_frames`, at which point we
give up holding and snap to the raw value so a genuinely long occlusion
doesn't leave a stale point frozen on screen forever.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


class LandmarkEMASmoother:
    def __init__(self, cfg: dict, num_landmarks: int = 33):
        self.alpha = cfg["smoothing"]["alpha"]
        self.max_hold_frames = cfg["smoothing"]["max_hold_frames"]
        self.min_visibility = cfg["min_landmark_visibility"]

        self._smoothed: Optional[np.ndarray] = None       # (N, 2)
        self._hold_counts = np.zeros(num_landmarks, dtype=np.int32)

    def reset(self) -> None:
        self._smoothed = None
        self._hold_counts[:] = 0

    def update(self, landmarks_xy: np.ndarray, visibility: np.ndarray) -> np.ndarray:
        """Return smoothed (N, 2) coords for this frame. Call once per frame,
        in order — this is stateful."""
        if self._smoothed is None:
            self._smoothed = landmarks_xy.copy()
            return self._smoothed.copy()

        visible = visibility >= self.min_visibility
        forced_snap = self._hold_counts >= self.max_hold_frames

        update_mask = visible | forced_snap
        hold_mask = ~visible & ~forced_snap

        # EMA update where visible (or forced), hold last value otherwise.
        new_smoothed = self._smoothed.copy()
        a = self.alpha
        new_smoothed[update_mask] = (
            a * landmarks_xy[update_mask] + (1 - a) * self._smoothed[update_mask]
        )
        # hold_mask rows: leave new_smoothed as the previous value (no-op)

        self._hold_counts[visible] = 0
        self._hold_counts[hold_mask] += 1
        self._hold_counts[forced_snap] = 0  # reset after snapping

        self._smoothed = new_smoothed
        return self._smoothed.copy()
