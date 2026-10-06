"""
Wires the pieces together: webcam frame -> pose -> smoothing -> features ->
punch detection -> scoring -> overlay. Intentionally thin — all the actual
logic lives in the modules it calls, so this file should stay easy to read
top-to-bottom as "here's the sequence of steps."
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from config import load_config
from pose_extraction import PoseExtractor
from smoothing import LandmarkEMASmoother
from features import FeatureTracker
from scoring import PunchDetector, score_punch


class LivePipeline:
    def __init__(self, config_path: str | None = None, camera_index: int = 0):
        self.cfg = load_config(config_path)
        self.extractor = PoseExtractor(self.cfg)
        self.smoother = LandmarkEMASmoother(self.cfg)
        self.feature_tracker = FeatureTracker()
        self.punch_detector = PunchDetector(self.cfg)
        self.camera_index = camera_index

        self._last_scores: dict[str, float] = {"left": 0.0, "right": 0.0}
        self._last_sub_scores: dict[str, dict] = {}

    def run(self, display: bool = True, video_path: str | None = None, output_path: str | None = None) -> None:
        """source = webcam (default, self.camera_index) or a recorded clip
        (video_path). Pass output_path to write an annotated copy to disk —
        useful when there's no display available, or you want to review
        footage after the fact instead of live."""
        is_file = video_path is not None
        cap = cv2.VideoCapture(video_path if is_file else self.camera_index)
        if not cap.isOpened():
            src_desc = video_path if is_file else f"camera index {self.camera_index}"
            raise RuntimeError(f"Could not open {src_desc}")

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                if not is_file:
                    frame = cv2.flip(frame, 1)  # mirror webcam for a natural feel; don't flip recorded clips

                # Recorded clips: use the file's own timestamps so angular
                # velocity is correct regardless of playback/processing speed.
                timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC) if is_file else time.time() * 1000.0

                self.process_frame(frame, timestamp_ms)

                if writer is not None:
                    writer.write(frame)
                if display:
                    cv2.imshow("Boxing Form Grader", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if display:
                cv2.destroyAllWindows()
            self.extractor.close()

    def process_frame(self, frame_bgr: np.ndarray, timestamp_ms: float) -> dict:
        """Runs one frame through the full pipeline and draws overlays onto
        frame_bgr in place. Returns a dict of the latest scores/features for
        callers (e.g. app/demo_ui.py, or tests) that want the raw numbers.

        Split out from run() so this is testable/embeddable without owning a
        camera loop.
        """
        pose_frame = self.extractor.process(frame_bgr, timestamp_ms)

        if not pose_frame.detected:
            self._draw_status(frame_bgr, "No pose detected")
            return {"detected": False}

        smoothed_xy = self.smoother.update(pose_frame.landmarks_xy, pose_frame.visibility)
        feats = self.feature_tracker.update(smoothed_xy, timestamp_ms)
        triggered_sides = self.punch_detector.check(feats)

        for side in triggered_sides:
            # Score peak swing velocity, not the instantaneous value at the
            # trigger frame (full extension = the arm has stopped moving).
            feats.wrist_angular_velocity_dps[side] = self.feature_tracker.pop_peak_velocity(side)
            result = score_punch(feats, self.cfg, side)
            self._last_scores[side] = result.composite
            self._last_sub_scores[side] = result.sub_scores

        if self.cfg["overlay"]["draw_skeleton"]:
            # Draw using the smoothed coords, not raw, so the overlay doesn't jitter.
            smoothed_pose_frame = pose_frame
            smoothed_pose_frame.landmarks_xy = smoothed_xy
            self.extractor.draw_landmarks(frame_bgr, smoothed_pose_frame)

        self._draw_scores(frame_bgr)

        return {
            "detected": True,
            "features": feats,
            "triggered_sides": triggered_sides,
            "scores": dict(self._last_scores),
        }

    def _draw_status(self, frame_bgr: np.ndarray, text: str) -> None:
        cv2.putText(frame_bgr, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    def _draw_scores(self, frame_bgr: np.ndarray) -> None:
        overlay_cfg = self.cfg["overlay"]
        good_thresh = overlay_cfg["score_good_threshold"]
        y = 40
        for side in ("left", "right"):
            score = self._last_scores[side]
            color = tuple(overlay_cfg["good_color_bgr"] if score >= good_thresh else overlay_cfg["bad_color_bgr"])
            text = f"{side.upper()}: {score:.0f}"
            cv2.putText(
                frame_bgr, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX,
                overlay_cfg["score_font_scale"], color, 2,
            )
            y += 40
