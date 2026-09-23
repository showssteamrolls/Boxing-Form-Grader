"""
Step 2 of calibration: reads the CSV(s) log_metrics.py produced and derives
min_good/max_good/tolerance for each metric, per punch type, from the actual
distribution of your reps — instead of the guessed defaults in config.py.

Usage:
    python app/calibrate_thresholds.py --csv data/calibration/jab_metrics.csv --punch-type jab
    python app/calibrate_thresholds.py --csv data/calibration/*.csv --punch-type cross

Prints a YAML block per punch type, ready to save as a config override and
pass to demo_ui.py / live_pipeline.py with --config.
"""
from __future__ import annotations

import argparse
import csv
import glob
import statistics
from pathlib import Path


METRICS = [
    "elbow_extension_deg",
    "hip_rotation_deg",
    "guard_height_ratio",
    "wrist_angular_velocity_dps",
]


def percentile(values: list[float], pct: float) -> float:
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    k = (len(values) - 1) * (pct / 100.0)
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    if lo == hi:
        return values[lo]
    frac = k - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def load_rows(csv_paths: list[str], punch_type: str) -> list[dict]:
    rows = []
    for path in csv_paths:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if row["punch_type"] == punch_type:
                    rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive thresholds from logged punch metrics")
    parser.add_argument("--csv", nargs="+", required=True, help="One or more CSV paths or glob patterns from log_metrics.py")
    parser.add_argument("--punch-type", required=True, choices=["jab", "cross"])
    parser.add_argument("--low-pct", type=float, default=10.0, help="Lower percentile for min_good (default 10th)")
    parser.add_argument("--high-pct", type=float, default=90.0, help="Upper percentile for max_good (default 90th)")
    args = parser.parse_args()

    csv_paths = []
    for pattern in args.csv:
        matched = glob.glob(pattern)
        csv_paths.extend(matched if matched else [pattern])

    rows = load_rows(csv_paths, args.punch_type)
    if len(rows) < 10:
        print(f"Only {len(rows)} rows found for punch_type={args.punch_type} — need at least 10-15 to calibrate meaningfully, ideally 25+.")
        return

    print(f"# {args.punch_type} — calibrated from {len(rows)} reps across {len(csv_paths)} file(s)")
    print("thresholds:")
    for metric in METRICS:
        values = [float(r[metric]) for r in rows]
        min_good = percentile(values, args.low_pct)
        max_good = percentile(values, args.high_pct)
        spread = max_good - min_good
        tolerance = max(spread * 0.5, 1e-3)

        print(f"  {metric}:")
        print(f"    min_good: {round(min_good, 2)}")
        print(f"    max_good: {round(max_good, 2)}")
        print(f"    tolerance: {round(tolerance, 2)}")
        print(f"    # observed range: {round(min(values), 2)} to {round(max(values), 2)}, "
              f"median {round(statistics.median(values), 2)}, n={len(values)}")


if __name__ == "__main__":
    main()
