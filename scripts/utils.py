"""Koch joint coordinates; no hardware access or Isaac/LeRobot dependency.

Last axis order: shoulder pan/lift, elbow flex, wrist flex/roll, gripper.
Zero counts are provisional user measurements, not fitted encoder scales.
Present_Position already contains Homing_Offset: never add it again here.
Gripper assumes the URDF driven joint rotates directly with the motor.
"""
import json
from pathlib import Path

import numpy as np

JOINT_NAMES = (
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
)
SIM_JOINT_NAMES = tuple(f"follower_joint{i}" for i in range(1, 6)) + ("follower_joint_gripper",)
ZERO_COUNTS = np.array([2105, 1615, 1768, 2630, 1405, 1447], dtype=np.float64)
DIRECTIONS = np.ones(6, dtype=np.float64)
RAD_PER_COUNT = 2.0 * np.pi / 4096.0
REFERENCE_HOMING_OFFSETS = (46, -986, -279, -416, 1471, -538)
DEFAULT_CALIBRATION = Path.home() / (
    ".cache/huggingface/lerobot/calibration/robots/koch_follower/koch11_follower.json"
)


def _joints(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 0 or values.shape[-1] != 6:
        raise ValueError("Expected joint values with shape (..., 6) in JOINT_NAMES order")
    if not np.isfinite(values).all():
        raise ValueError("Joint values must be finite")
    return values


def load_calibration(path=DEFAULT_CALIBRATION):
    """Read ranges and reject calibration changes that invalidate measured zeros."""
    calibration = json.loads(Path(path).expanduser().read_text())
    for index, name in enumerate(JOINT_NAMES):
        entry = calibration[name]
        limits = np.array([entry["range_min"], entry["range_max"]], dtype=float)
        if not np.isfinite(limits).all() or limits[1] <= limits[0]:
            raise ValueError(f"Invalid calibration range for {name}")
        if entry["id"] != index + 1:
            raise ValueError(f"Motor ID changed for {name}; verify the joint mapping")
        if entry["homing_offset"] != REFERENCE_HOMING_OFFSETS[index] or entry["drive_mode"] != 0:
            raise ValueError(f"Calibration changed for {name}; recheck ZERO_COUNTS and DIRECTIONS")
    return calibration


def _ranges(calibration):
    low = np.array([calibration[name]["range_min"] for name in JOINT_NAMES], dtype=float)
    high = np.array([calibration[name]["range_max"] for name in JOINT_NAMES], dtype=float)
    if not np.isfinite([low, high]).all() or np.any(high <= low):
        raise ValueError("Calibration ranges must be finite and increasing")
    return low, high


def encoder_to_sim(counts):
    """Reported motor counts -> absolute URDF radians; no clipping or wrapping."""
    return (_joints(counts) - ZERO_COUNTS) * DIRECTIONS * RAD_PER_COUNT


def sim_to_encoder(joint_radians, *, round_counts=False):
    """URDF radians -> equivalent counts, optionally rounded to encoder ticks."""
    counts = ZERO_COUNTS + _joints(joint_radians) / (DIRECTIONS * RAD_PER_COUNT)
    return np.rint(counts).astype(np.int64) if round_counts else counts


def encoder_to_lerobot(counts, calibration, *, use_degrees=False):
    """Motor counts -> LeRobot state, before checkpoint mean/std preprocessing.

    Default arms [-100, 100], gripper [0, 100], with LeRobot range clipping.
    use_degrees affects arms only and follows LeRobot's 4095-based convention;
    that software convention is distinct from the physical 4096 counts/rev.
    """
    counts = _joints(counts)
    low, high = _ranges(calibration)
    fraction = (np.clip(counts, low, high) - low) / (high - low)
    values = fraction * 200.0 - 100.0
    values[..., 5] = fraction[..., 5] * 100.0
    if use_degrees:
        values[..., :5] = (counts[..., :5] - ((low + high) / 2.0)[:5]) * 360.0 / 4095.0
    return values


def lerobot_to_encoder(values, calibration, *, use_degrees=False):
    """Postprocessed LeRobot actions -> integer counts, matching bus truncation."""
    values = _joints(values)
    low, high = _ranges(calibration)
    fraction = (np.clip(values, -100.0, 100.0) + 100.0) / 200.0
    fraction[..., 5] = np.clip(values[..., 5], 0.0, 100.0) / 100.0
    counts = low + fraction * (high - low)
    if use_degrees:
        counts[..., :5] = values[..., :5] * 4095.0 / 360.0 + ((low + high) / 2.0)[:5]
    return np.trunc(counts).astype(np.int64)


def sim_to_lerobot(joint_radians, calibration, *, use_degrees=False):
    """Simulated joint state -> ACT observation.state in LeRobot motor units."""
    return encoder_to_lerobot(sim_to_encoder(joint_radians), calibration, use_degrees=use_degrees)


def lerobot_to_sim(values, calibration, *, use_degrees=False):
    """ACT action AFTER saved postprocessing -> absolute simulation radians."""
    return encoder_to_sim(lerobot_to_encoder(values, calibration, use_degrees=use_degrees))


def joint_report(joint_radians, calibration, *, use_degrees=False):
    """JSON-ready diagnostics; expose range violations before normalization clips."""
    radians = _joints(joint_radians)
    counts = sim_to_encoder(radians)
    low, high = _ranges(calibration)
    return {
        "joint_order": list(JOINT_NAMES),
        "sim_radians": radians.tolist(),
        "encoder_counts": counts.tolist(),
        "lerobot_state": sim_to_lerobot(radians, calibration, use_degrees=use_degrees).tolist(),
        "outside_calibration_range": ((counts < low) | (counts > high)).tolist(),
        "lerobot_arm_units": "degrees" if use_degrees else "range_m100_100",
        "calibration_status": "provisional measured zeros; shoulder lift/elbow/wrist flex need recheck",
    }
