"""Validated manual grasp waypoints and timing; no simulator dependency."""
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np

from utils import encoder_to_sim, lerobot_to_sim

PHASES = ("open", "align", "close", "lift")
LOWER = np.array([-2.7] * 5 + [0.0])
UPPER = np.array([2.7] * 5 + [1.5])


@dataclass
class GraspSequence:
    poses: np.ndarray
    move_seconds: float
    hold_seconds: float

    @property
    def duration(self):
        return len(PHASES) * (self.move_seconds + self.hold_seconds)

    def sample(self, elapsed, initial):
        """Interpolate from the measured reset pose; hold each reached waypoint."""
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("Sequence time must be finite and nonnegative")
        interval = self.move_seconds + self.hold_seconds
        index = min(int(elapsed / interval), len(PHASES) - 1)
        start = np.asarray(initial) if index == 0 else self.poses[index - 1]
        fraction = np.clip((elapsed - index * interval) / self.move_seconds, 0, 1)
        return PHASES[index], start + fraction * (self.poses[index] - start)


def load_sequence(path, calibration, *, use_degrees=False):
    data = json.loads(Path(path).expanduser().read_text())
    if not isinstance(data, dict):
        raise ValueError("Grasp sequence must be a JSON object")
    if data.get("units") not in ("radians", "encoder", "lerobot"):
        raise ValueError("Sequence units must be radians, encoder, or lerobot")
    poses = []
    for phase in PHASES:
        values = data.get(phase)
        if not isinstance(values, list) or len(values) != 6 or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) for v in values
        ):
            raise ValueError(f"Fill {phase} with six numeric joint values (no null placeholders)")
        pose = np.asarray(values, dtype=float)
        if not np.isfinite(pose).all():
            raise ValueError(f"{phase} joint values must be finite")
        if data["units"] == "encoder":
            pose = encoder_to_sim(pose)
        elif data["units"] == "lerobot":
            # Reject out-of-range values rather than silently clipping the test.
            if not 0 <= pose[-1] <= 100 or (not use_degrees and np.any(np.abs(pose[:5]) > 100)):
                raise ValueError(f"{phase} is outside LeRobot action ranges")
            pose = lerobot_to_sim(pose, calibration, use_degrees=use_degrees)
        if np.any(pose < LOWER) or np.any(pose > UPPER):
            raise ValueError(f"{phase} exceeds simulation joint limits: {pose.tolist()}")
        poses.append(pose)
    poses = np.stack(poses)
    if not np.isclose(poses[0, -1], poses[1, -1]):
        raise ValueError("open and align must use the same gripper angle")
    if not np.allclose(poses[1, :5], poses[2, :5]):
        raise ValueError("align and close must use the same five arm angles")
    if not np.isclose(poses[2, -1], poses[3, -1]):
        raise ValueError("close and lift must use the same gripper angle")
    durations = []
    for key, default in (("move_seconds", 2.0), ("hold_seconds", 1.0)):
        value = data.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive")
        durations.append(float(value))
    return GraspSequence(poses, *durations)
