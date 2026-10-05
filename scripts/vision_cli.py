#!/usr/bin/env python3
"""Calibrate or inspect the interchangeable CV module without launching Isaac."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from koch_isaac.vision import ColorPlaneEstimator, PlaneCalibration, load_config
from koch_isaac.vision.color_plane import DEFAULT_CONFIG


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    calibrate = sub.add_parser("calibrate", help="Fit pixels -> measured workspace XY")
    calibrate.add_argument("--points", type=Path, required=True)
    calibrate.add_argument("--output", type=Path, required=True)
    inspect = sub.add_parser("inspect", help="Detect pink box in a saved RGB image")
    inspect.add_argument("--image", type=Path, required=True)
    inspect.add_argument("--calibration", type=Path, required=True)
    inspect.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    inspect.add_argument("--overlay", type=Path, help="Optional annotated image; no files by default")
    args = parser.parse_args()

    if args.command == "calibrate":
        points = json.loads(args.points.read_text())
        calibration = PlaneCalibration.from_points(**points)
        estimates = np.array([calibration.project(uv)[:2] for uv in points["pixels_uv"]])
        errors = np.linalg.norm(estimates - np.asarray(points["plane_xy_m"]), axis=-1)
        # Do not overwrite a user's measured calibration accidentally.
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(calibration.as_dict(), stream, indent=2)
        print(json.dumps({"calibration": str(args.output), "fit_errors_m": errors.tolist(),
                          "note": "Fit error is not validation accuracy; test held-out measured points."}))
        return

    bgr = cv2.imread(str(args.image))
    if bgr is None:
        parser.error("Cannot read image: " + str(args.image))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    estimator = ColorPlaneEstimator(PlaneCalibration.load(args.calibration), load_config(args.config))
    result = estimator.estimate(rgb)
    print(json.dumps(result.as_dict(), indent=2))
    if args.overlay:
        if args.overlay.exists():
            parser.error("Overlay already exists: " + str(args.overlay))
        overlay = rgb.copy()
        mask = estimator.mask(rgb).astype(bool)
        overlay[mask] = (0.5 * overlay[mask] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
        if result.pixel_uv:
            cv2.drawMarker(overlay, tuple(round(v) for v in result.pixel_uv), (255, 255, 0),
                           cv2.MARKER_CROSS, 15, 2)
        cv2.putText(overlay, result.reason, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        args.overlay.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.overlay), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)):
            raise OSError("Failed to write overlay")


if __name__ == "__main__":
    main()
