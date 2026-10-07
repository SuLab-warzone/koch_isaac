"""Grasp-first residual rewards; simulator signals never enter the frozen ACT."""
import torch
from isaaclab.managers import ManagerTermBase
from isaaclab.utils.math import matrix_from_quat, quat_apply_inverse
from .observations import box_position, box_com_position, tcp_position, goal_position
from .terminations import placement_candidate
from .. import settings as s
from ..reward_shaping import GraspShapingSettings, GraspRewardState, dense_signals


def _tensor(value):
    return value.torch if hasattr(value, "torch") else value


def grasp_snapshot(env, settings):
    robot, box = env.scene["robot"], env.scene["box"]
    body = robot.find_bodies("follower_gripper_static_1")[0][0]
    quat = _tensor(robot.data.body_quat_w)[:, body]
    local_error = quat_apply_inverse(quat, box_com_position(env) - tcp_position(env))
    gripper = robot.joint_names.index("follower_joint_gripper")
    angle = _tensor(robot.data.joint_pos)[:, gripper]
    rotation = matrix_from_quat(quat)
    box_rotation = matrix_from_quat(_tensor(box.data.root_quat_w))
    # Encourage a vertical approach and jaw axis parallel to either box edge.
    # Both signs and both rectangular edges are valid; exact pad geometry still
    # needs calibration. This is a TCP pose proxy, not a measured jaw aperture.
    closing_axis = rotation[:, :, 0]
    edge_alignment = torch.stack([
        (closing_axis * box_rotation[:, :, edge]).sum(-1).square() for edge in (0, 1)
    ], dim=-1).amax(-1)
    orientation = rotation[:, 2, 2].square() * edge_alignment
    half = local_error.new_tensor(s.BOX_SIZE) / 2
    extent = torch.matmul(box_rotation.abs(), half)
    table_top = s.TABLE_POS[2] + s.TABLE_SIZE[2] / 2
    bottom_height = box_position(env)[:, 2] - extent[:, 2] - table_top
    forces = []
    for name in ("static_finger_contact", "moving_finger_contact"):
        force = env.scene[name].data.force_matrix_w
        if force is None:
            raise RuntimeError("Grasp rewards require box-filtered finger contact sensors")
        pairs = _tensor(force).reshape(env.num_envs, -1, 3)
        if pairs.shape[1] != 1:
            raise RuntimeError("Expected one finger/box pair per reward contact sensor")
        forces.append(pairs[:, 0])
    goal_distance = torch.linalg.vector_norm(box_position(env) - goal_position(env), dim=-1)
    return dense_signals(local_error, orientation, angle, torch.stack(forces, dim=1),
                         bottom_height, goal_distance, settings)


class GraspReward(ManagerTermBase):
    """Share milestone state across dense terms without relying on delayed metrics."""
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if not hasattr(env, "_koch_grasp_reward_state"):
            env._koch_grasp_reward_state = GraspRewardState(
                env.num_envs, env.device, env.step_dt, GraspShapingSettings()
            )
        self.state = env._koch_grasp_reward_state

    def reset(self, env_ids=None):
        self.state.reset(env_ids)

    def __call__(self, env, component):
        if env.common_step_counter != self.state.last_step:
            signals = grasp_snapshot(env, self.state.settings)
            # Isaac computes terminations before rewards, before automatic reset.
            # Use the 0.5-second stable placement termination, not an instantaneous
            # placement candidate or EvaluatedKochEnv's post-reward metrics.
            success = env.termination_manager.get_term("success")
            self.state.update(env.common_step_counter, signals, success)
        return self.state.signals[component]


def reach_box(env, std=0.02):
    """Reward proximity of the configured TCP to the box's COM, not its link origin.

    Both points are environment-relative and measured in metres. std is the
    distance scale of the tanh reward, not a desired clearance above the box.
    The current centered cuboid has coincident COM and root, so this change alone
    does not lower a grasp; TCP_OFFSET must still describe the actual grasp point.
    """
    distance = torch.linalg.vector_norm(box_com_position(env) - tcp_position(env), dim=-1)
    return 1.0 - torch.tanh(distance / std)


def lift_box(env):
    """Stateless compatibility helper; configured terms use shared GraspReward."""
    return grasp_snapshot(env, GraspShapingSettings())["lift"]


def move_to_bin(env, std=0.08):
    return grasp_snapshot(env, GraspShapingSettings(transport_std=std))["transport"]


def placed(env):
    return placement_candidate(env).float()
