"""
Minimal entrypoint: runs the live pipeline against your default webcam.

Usage:
    python app/demo_ui.py                             # live webcam
    python app/demo_ui.py --camera 1                   # different camera
    python app/demo_ui.py --config path/to/thresholds.yaml
    python app/demo_ui.py --video clip.mp4              # grade a recorded clip instead
    python app/demo_ui.py --video clip.mp4 --output graded.mp4 --no-display
                                                          # save annotated output, no window
                                                          # (e.g. running on a machine with no display)

Press 'q' in the video window to quit (live/display mode only).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running this script directly without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from live_pipeline import LivePipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time boxing form grader")
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML thresholds override file")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default 0), ignored if --video is set")
    parser.add_argument("--video", type=str, default=None, help="Path to a recorded clip to grade instead of the live webcam")
    parser.add_argument("--output", type=str, default=None, help="Path to write an annotated copy of the input (requires --video)")
    parser.add_argument("--no-display", action="store_true", help="Don't open a video window (useful for headless runs with --output)")
    args = parser.parse_args()

    pipeline = LivePipeline(config_path=args.config, camera_index=args.camera)
    mode = f"clip {args.video}" if args.video else "live webcam"
    print(f"Starting boxing form grader on {mode} — press 'q' to quit." if not args.no_display else f"Starting boxing form grader on {mode}.")
    pipeline.run(display=not args.no_display, video_path=args.video, output_path=args.output)


if __name__ == "__main__":
    main()
