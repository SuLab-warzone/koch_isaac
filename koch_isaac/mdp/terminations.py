"""Containment uses all oriented box corners, rest height, velocity and release.

Success must persist for 0.5 s. This is a starter metric; validate against real trials.
"""
import torch
from isaaclab.managers import ManagerTermBase
from isaaclab.utils.math import matrix_from_quat
from .observations import box_position, tcp_position
from .. import settings as s


def placement_candidate(env):
    box = env.scene["box"]
    pos = box_position(env)
    half = torch.tensor(s.BOX_SIZE, device=env.device)/2
    extent = torch.matmul(matrix_from_quat(box.data.root_quat_w.torch).abs(), half)
    center = torch.tensor(s.BIN_POS[:2], device=env.device)
    inner = torch.tensor(s.BIN_INNER[:2], device=env.device)/2
    inside = ((pos[:, :2]-center).abs()+extent[:, :2] < inner-0.001).all(-1)
    resting = (pos[:, 2]-extent[:, 2]-s.BIN_FLOOR).abs() < 0.004
    below_rim = pos[:, 2]+extent[:, 2] < s.BIN_FLOOR+s.BIN_INNER[2]+0.001
    still = (torch.linalg.vector_norm(box.data.root_lin_vel_w.torch, dim=-1)<0.025) & (torch.linalg.vector_norm(box.data.root_ang_vel_w.torch, dim=-1)<0.3)
    released = torch.linalg.vector_norm(tcp_position(env)-pos, dim=-1)>0.055
    return inside & resting & below_rim & still & released


class PlacementSuccess(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def reset(self, env_ids=None):
        self.count[slice(None) if env_ids is None else env_ids] = 0

    def __call__(self, env, hold_seconds=0.5):
        self.count[:] = torch.where(placement_candidate(env), self.count+1, 0)
        return self.count >= round(hold_seconds/env.step_dt)


def box_lost(env):
    p = box_position(env)
    return (p[:, 2] < -0.08) | (p[:, 0].abs()>0.65) | (p[:, 1].abs()>0.5)
