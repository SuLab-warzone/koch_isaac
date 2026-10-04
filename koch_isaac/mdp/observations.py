"""State observations are for an RL baseline, not direct ACT/Diffusion input."""
import torch
from isaaclab.utils.math import quat_apply
from .. import settings as s


def box_position(env):
    return env.scene["box"].data.root_pos_w.torch - env.scene.env_origins


def goal_position(env):
    return torch.tensor((s.BIN_POS[0], s.BIN_POS[1], s.BIN_FLOOR+s.BOX_SIZE[2]/2),
                        device=env.device).expand(env.num_envs, -1)


def tcp_position(env):
    robot = env.scene["robot"]
    idx = robot.find_bodies("follower_gripper_static_1")[0][0]
    offset = torch.tensor(s.TCP_OFFSET, device=env.device).expand(env.num_envs, -1)
    return robot.data.body_pos_w.torch[:, idx] + quat_apply(robot.data.body_quat_w.torch[:, idx], offset) - env.scene.env_origins


def front_rgb(env):
    return env.scene["front_camera"].data.output["rgb"][..., :3].to(torch.float32) / 255.0
