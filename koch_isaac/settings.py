"""Provisional scene measurements [m, kg, rad]; replace with real measurements.

World: +X is in front of the robot, +Y is left, +Z is up.
Table top is z=0; all poses below are relative to an environment origin.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / "assets/koch/koch_follower.urdf"
JOINT_NAMES = [f"follower_joint{i}" for i in range(1, 6)] + ["follower_joint_gripper"]
HOME = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
TABLE_SIZE = (0.32, 0.44, 0.04)
TABLE_POS = (0.09, 0.0, -0.02)
BOX_SIZE = (0.030, 0.020, 0.008)
BOX_MASS = 0.010
BOX_POS = (0.13, -0.02, BOX_SIZE[2] / 2 + 0.002)
BIN_POS = (0.17, -0.16, 0.0)
BIN_INNER = (0.08, 0.06, 0.05)
BIN_WALL = 0.002
BIN_FLOOR = 0.002
# URDF gripper frame -> approximate center between fingertips [m].
TCP_OFFSET = (0.0, 0.0, -0.053)
CAMERA_EYE = (0.08, 0.5, 0.3)
CAMERA_TARGET = (0.10, 0.000, -0.015)
