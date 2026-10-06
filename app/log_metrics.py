"""
Runs a clip through the tracking pipeline and logs the raw metric values for
every detected punch to a CSV — no scoring, just the numbers.

This is step 1 of calibration: turn footage into a data table. Run this
against a clip of clean, deliberate reps of ONE punch type (jab or cross —
don't mix them in a single clip, since they're biomechanically different and
need separate threshold ranges).

Usage:
    python app/log_metrics.py --video clip.mp4 --punch-type jab --output data/calibration/jab_metrics.csv
    python app/log_metrics.py --video cross_clip.mp4 --punch-type cross --output data/calibration/cross_metrics.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2  # noqa: E402

from config import load_config  # noqa: E402
from pose_extraction import PoseExtractor  # noqa: E402
from smoothing import LandmarkEMASmoother  # noqa: E402
from features import FeatureTracker  # noqa: E402
from scoring import PunchDetector  # noqa: E402


FIELDNAMES = [
    "source_video",
    "timestamp_s",
    "side",
    "punch_type",
    "elbow_extension_deg",
    "hip_rotation_deg",
    "guard_height_ratio",
    "wrist_angular_velocity_dps",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Log raw per-punch metrics for threshold calibration")
    parser.add_argument("--video", required=True, help="Path to a clip of clean, deliberate reps")
    parser.add_argument("--punch-type", required=True, choices=["jab", "cross"], help="All punches in this clip must be this type")
    parser.add_argument("--output", required=True, help="CSV path to write (created if missing, rows appended if it exists)")
    parser.add_argument("--config", default=None, help="Optional YAML threshold override, same as demo_ui.py")
    args = parser.parse_args()

    cfg = load_config(args.config)
    extractor = PoseExtractor(cfg)
    smoother = LandmarkEMASmoother(cfg)
    tracker = FeatureTracker()
    detector = PunchDetector(cfg)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {args.video}")

    rows = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            pose_frame = extractor.process(frame, timestamp_ms)
            if not pose_frame.detected:
                continue

            smoothed_xy = smoother.update(pose_frame.landmarks_xy, pose_frame.visibility)
            feats = tracker.update(smoothed_xy, timestamp_ms)
            triggered_sides = detector.check(feats)

            for side in triggered_sides:
                other_side = "left" if side == "right" else "right"
                # Peak velocity during the swing, not the instantaneous value
                # at the trigger frame — detection fires at full extension,
                # where the arm has already stopped moving.
                peak_velocity_dps = tracker.pop_peak_velocity(side)
                rows.append({
                    "source_video": Path(args.video).name,
                    "timestamp_s": round(timestamp_ms / 1000.0, 3),
                    "side": side,
                    "punch_type": args.punch_type,
                    "elbow_extension_deg": round(feats.elbow_extension_deg[side], 2),
                    "hip_rotation_deg": round(feats.hip_rotation_deg, 2),
                    "guard_height_ratio": round(feats.guard_height_ratio[other_side], 3),
                    "wrist_angular_velocity_dps": round(peak_velocity_dps, 1),
                })
    finally:
        cap.release()
        extractor.close()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_path.exists()

    with open(output_path, "a" if file_exists else "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)

    action = "Appended" if file_exists else "Wrote"
    print(f"{action} {len(rows)} punches ({args.punch_type}) to {output_path}")
    if len(rows) < 15:
        print(f"Only {len(rows)} punches detected — aim for 25+ per punch type before calibrating.")


if __name__ == "__main__":
    main()
