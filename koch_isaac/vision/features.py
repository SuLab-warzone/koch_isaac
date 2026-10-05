"""Per-environment planar-cue gating, usable with simulation or real robot state."""
import numpy as np


class PlanarBoxFeatures:
    """Expose box-minus-TCP plus a validity flag, never hidden simulator box state.

    Projection cannot detect height from RGB alone. Conservatively retire the cue
    for the rest of an episode once the gripper closes near the estimated box.
    This is a grasp-ATTEMPT heuristic, not proof of a grasp. It deliberately cannot
    support a second vision-guided retry after that attempt; use a 3D estimator
    for lifting, retries, or a box that can move vertically before grasping.
    """
    def __init__(self, num_envs, config):
        self.config = config
        self.disabled = np.zeros(num_envs, dtype=bool)
        self.positions = np.zeros((num_envs, 3), dtype=np.float32)
        self.valid = np.zeros(num_envs, dtype=bool)
        self.tcp = np.zeros((num_envs, 3), dtype=np.float32)

    def reset(self, ids=None):
        ids = slice(None) if ids is None else ids
        self.disabled[ids] = False
        self.valid[ids] = False
        self.positions[ids] = 0
        self.tcp[ids] = 0

    def note_gripper(self, gripper_angle):
        """Called with both measured and final commanded angle; small angle=closed."""
        near = np.linalg.norm(self.positions - self.tcp, axis=-1) < self.config["grasp_gate_distance_m"]
        closing = np.asarray(gripper_angle) <= self.config["grasp_gate_angle_rad"]
        self.disabled |= self.valid & near & closing

    def update(self, estimates, tcp, measured_gripper):
        self.tcp[:] = tcp
        self.valid[:] = [estimate.valid for estimate in estimates]
        self.positions[:] = [estimate.position for estimate in estimates]
        self.note_gripper(measured_gripper)
        usable = self.valid & ~self.disabled
        result = np.zeros((len(estimates), 4), dtype=np.float32)
        result[usable, :3] = (self.positions[usable] - self.tcp[usable]) / self.config["position_scale_m"]
        result[:, 3] = usable
        return result
