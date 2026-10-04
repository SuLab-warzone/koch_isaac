"""Small starter reward; tune the task here after calibrating geometry."""
import torch
from .observations import box_position, tcp_position, goal_position
from .terminations import placement_candidate
from .. import settings as s


def reach_box(env, std=0.06):
    return 1.0 - torch.tanh(torch.linalg.vector_norm(box_position(env)-tcp_position(env), dim=-1)/std)


def lift_box(env):
    return (box_position(env)[:, 2] > s.BOX_SIZE[2]/2 + 0.025).float()


def move_to_bin(env, std=0.08):
    distance = torch.linalg.vector_norm(box_position(env)-goal_position(env), dim=-1)
    return (1.0-torch.tanh(distance/std))*lift_box(env)


def placed(env):
    return placement_candidate(env).float()
