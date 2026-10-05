"""Pink colour segmentation plus plane projection; no learned model or labels."""
import json
from pathlib import Path

import cv2
import numpy as np

from .base import BoxEstimate

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config/vision_color_plane.json"


def load_config(path=DEFAULT_CONFIG):
    cfg = json.loads(Path(path).read_text())
    lo, hi = np.asarray(cfg["hsv_lower"]), np.asarray(cfg["hsv_upper"])
    if (lo.shape != (3,) or hi.shape != (3,) or
            not np.isfinite(lo).all() or not np.isfinite(hi).all() or
            (lo < 0).any() or (hi > [179, 255, 255]).any() or (lo > hi).any()):
        raise ValueError("HSV bounds must be ordered OpenCV values: H 0..179, S/V 0..255")
    if not 0 < cfg["min_area_px"] < cfg["max_area_px"]:
        raise ValueError("Invalid component area bounds")
    if not 0 < cfg["ambiguity_ratio"] <= 1:
        raise ValueError("ambiguity_ratio must be in (0,1]")
    if not 0 <= cfg["projection_height_fraction"] <= 1:
        raise ValueError("projection_height_fraction must be in [0,1]")
    workspace = np.asarray(cfg["workspace_xy_m"])
    if workspace.shape != (2, 2) or not np.isfinite(workspace).all() or (workspace[:, 0] >= workspace[:, 1]).any():
        raise ValueError("workspace_xy_m must contain increasing X and Y bounds")
    if any(not np.isfinite(cfg[key]) or cfg[key] <= 0
           for key in ("position_scale_m", "grasp_gate_distance_m", "grasp_gate_angle_rad")):
        raise ValueError("Feature scale and grasp gate thresholds must be positive")
    return cfg


class ColorPlaneEstimator:
    """Stateless estimate from one RGB frame; disappearance returns invalid immediately.

    The visible pink silhouette is an approximation to the box centre, so
    perspective, side faces and partial occlusion can introduce several mm of error.
    This is a planar approach cue, not a full 3D pose tracker or a grasp detector.
    """
    def __init__(self, calibration, config):
        self.calibration = calibration
        self.config = config

    def mask(self, rgb):
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("Expected uint8 RGB [height,width,3]")
        if (rgb.shape[1], rgb.shape[0]) != self.calibration.image_size:
            raise ValueError("Image size differs from camera calibration")
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        return cv2.inRange(hsv, np.array(self.config["hsv_lower"], np.uint8),
                          np.array(self.config["hsv_upper"], np.uint8))

    def estimate(self, rgb):
        mask = self.mask(rgb)
        n, _, stats, centres = cv2.connectedComponentsWithStats(mask, connectivity=8)
        candidates = []
        bounds = np.asarray(self.config["workspace_xy_m"])
        for label in range(1, n):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if not self.config["min_area_px"] <= area <= self.config["max_area_px"]:
                continue
            uv = tuple(float(v) for v in centres[label])
            position = self.calibration.project(uv)
            if position is None or not ((position[:2] >= bounds[:, 0]) &
                                       (position[:2] <= bounds[:, 1])).all():
                continue
            candidates.append((area, uv, position))
        if not candidates:
            return BoxEstimate.missing("no_component_in_workspace")
        candidates.sort(key=lambda item: item[0], reverse=True)
        area, uv, position = candidates[0]
        # Similar-size pink objects are ambiguous: do not silently jump to one.
        if len(candidates) > 1 and candidates[1][0] >= area * self.config["ambiguity_ratio"]:
            return BoxEstimate.missing("ambiguous_components", uv, area)
        return BoxEstimate(position, True, "plane_assumption", uv, area)
