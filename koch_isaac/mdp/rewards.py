"""Small starter reward; tune the task here after calibrating geometry."""
import torch
from .observations import box_position, box_com_position, tcp_position, goal_position
from .terminations import placement_candidate
from .. import settings as s


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
    return (box_position(env)[:, 2] > s.BOX_SIZE[2]/2 + 0.025).float()


def move_to_bin(env, std=0.08):
    distance = torch.linalg.vector_norm(box_position(env)-goal_position(env), dim=-1)
    return (1.0-torch.tanh(distance/std))*lift_box(env)


def placed(env):
    return placement_candidate(env).float()
