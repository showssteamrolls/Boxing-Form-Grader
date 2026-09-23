# Boxing Form Grader

Real-time pose-estimation tool that grades boxing punch form from a webcam
feed, using MediaPipe's PoseLandmarker.

## What it grades (per punch)

| Metric | What it measures |
|---|---|
| Elbow extension | How straight the punching arm is at full extension |
| Hip rotation | Torso twist relative to the shoulder line (power generation) |
| Guard height | Non-punching wrist height relative to shoulder (are you dropping your guard) |
| Punch angular velocity | How fast the arm snaps out (deg/sec) |

Each metric is graded against a tunable `[min_good, max_good]` range with a
linear falloff tolerance band, then combined into a single 0–100 composite
score, weighted per metric. All of this lives in `src/config.py` — nothing
else in the codebase hardcodes a threshold.

## Setup

```bash
pip install -r requirements.txt
python app/demo_ui.py
```

On first run, the pose model (`pose_landmarker_lite.task`, ~5–10MB) is
auto-downloaded to `models/` and cached — this needs one working internet
connection the first time only.

Press `q` in the video window to quit.

### Options

```bash
python app/demo_ui.py --camera 1                 # use a different camera
python app/demo_ui.py --config my_thresholds.yaml  # override thresholds
```

### Note on MediaPipe API version

This project uses MediaPipe's current **Tasks API**
(`mediapipe.tasks.python.vision.PoseLandmarker`). If you've seen older
tutorials using `mp.solutions.pose.Pose(...)` — that legacy API was removed
from MediaPipe releases 0.10.9+ and will raise `AttributeError: module
'mediapipe' has no attribute 'solutions'` if you try it on a current install.
Everything here is already written against the current API.

## Retuning thresholds

Copy the `thresholds` and `score_weights` blocks out of `src/config.py`'s
`DEFAULT_CONFIG` into a YAML file, edit the numbers, and pass it with
`--config`. Example (`my_thresholds.yaml`):

```yaml
thresholds:
  elbow_extension_deg:
    min_good: 160
    max_good: 180
    tolerance: 20
score_weights:
  elbow_extension_deg: 0.4
  hip_rotation_deg: 0.3
  guard_height_ratio: 0.2
  punch_angular_velocity_dps: 0.1
```

`score_weights` must sum to 1.0 — this is validated on load and will raise a
clear error if it doesn't. Use `notebooks/threshold_tuning.ipynb` to plot
recorded feature values against your current thresholds and dial them in
against real footage instead of guessing.

## How occlusion is handled

MediaPipe still emits a coordinate for a landmark it can't really see (e.g.
a glove crossing in front of the torso) — it's a low-confidence guess, not a
missing value. `src/smoothing.py` gates the EMA update on each landmark's
per-frame visibility score: below `min_landmark_visibility` (config), the
last trusted position is held rather than smoothed-in, so a brief occlusion
doesn't drag the tracked point through a wrong location. If an occlusion
runs longer than `max_hold_frames`, the smoother gives up holding and snaps
to the raw value so a stale point doesn't stick on screen indefinitely.

## Repo structure

```
boxing-form-grader/
├── README.md
├── requirements.txt
├── data/raw_footage/          # self-recorded clips for offline tuning
├── models/                    # auto-downloaded pose model (gitignored)
├── src/
│   ├── config.py              # all tunable thresholds/weights — single source of truth
│   ├── pose_extraction.py     # MediaPipe PoseLandmarker → keypoints + visibility
│   ├── features.py            # joint angles, angular velocity, guard height
│   ├── smoothing.py           # visibility-gated EMA smoothing
│   ├── scoring.py             # thresholds → per-metric + composite score
│   └── live_pipeline.py       # webcam capture → overlay → live score
├── notebooks/
│   └── threshold_tuning.ipynb
├── app/
│   └── demo_ui.py             # webcam demo entrypoint
└── tests/
    └── test_features.py       # pure-math regression tests (no camera needed)
```

## Running tests

```bash
pip install pytest
pytest tests/ -v
```

These test the angle/velocity math in isolation with synthetic landmark
arrays — no camera or MediaPipe model required, so they run in under a
second and won't break when you retune thresholds.
