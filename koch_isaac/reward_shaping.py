"""Vectorized grasp shaping and episode milestones, independent of Isaac APIs."""
from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class GraspShapingSettings:
    # Local TCP errors [m]; Z is tight because the box is only 8 mm tall.
    alignment_std: tuple = (0.012, 0.008, 0.004)
    reach_std: float = 0.02
    near_distance: float = 0.04
    contact_threshold: float = 0.01
    contact_target: float = 0.08
    opposing_cosine: float = -0.25
    # Provisional direct-drive angles, not jaw width. Override after calibration.
    open_angle: float = 1.0
    closed_angle: float = 0.0
    aperture_std: float = 0.25
    lift_target: float = 0.04
    milestone_height: float = 0.025
    milestone_hold: float = 0.2
    transport_std: float = 0.08


def dense_signals(local_error, orientation_score, gripper_angle, finger_forces,
                  bottom_height, goal_distance, settings):
    """Return bounded dense signals and a physically gated grasp candidate.

    Forces are [env, static/moving, xyz] box-filtered normal resultants. A single
    finger, nonopposing contacts, or a box launched into the air gets no lift.
    """
    std = local_error.new_tensor(settings.alignment_std)
    alignment = torch.exp(-0.5 * (local_error / std).square().sum(-1)) * orientation_score.clamp(0, 1)
    distance = torch.linalg.vector_norm(local_error, dim=-1)
    reach = 1 - torch.tanh(distance / settings.reach_std)
    normal = torch.linalg.vector_norm(finger_forces, dim=-1)
    cosine = (finger_forces[:, 0] * finger_forces[:, 1]).sum(-1) / (normal.prod(-1).clamp_min(1e-9))
    dual = ((normal > settings.contact_threshold).all(-1)
            & (cosine < settings.opposing_cosine) & (distance < settings.near_distance))
    contact = (normal.amin(-1) / settings.contact_target).clamp(0, 1) * dual.float()

    # Blend opening/closing with alignment, but stop asking for a smaller angle
    # once opposing jaw contact exists. Closing on air far away earns nothing.
    desired = settings.open_angle + alignment * (settings.closed_angle - settings.open_angle)
    aperture = torch.exp(-0.5 * ((gripper_angle - desired) / settings.aperture_std).square())
    aperture = torch.where(dual, torch.ones_like(aperture), aperture) * reach
    lift = (bottom_height / settings.lift_target).clamp(0, 1) * dual.float()
    transport = (1 - torch.tanh(goal_distance / settings.transport_std)) * dual.float()
    return {"reach": reach, "alignment": alignment, "aperture": aperture,
            "contact": contact, "lift": lift, "transport": transport,
            "grasp_candidate": dual & (bottom_height >= settings.milestone_height)}


class GraspRewardState:
    """One update per control step, shared across reward terms; selective reset."""

    def __init__(self, num_envs, device, dt, settings):
        self.dt = dt
        self.settings = settings
        self.hold_frames = max(1, math.ceil(settings.milestone_hold / dt))
        self.count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.grasped = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.placed = torch.zeros_like(self.grasped)
        self.last_step = None
        self.signals = {}

    def update(self, step, signals, placement_success):
        if step == self.last_step:
            return self.signals
        self.last_step = step
        self.count = torch.where(signals["grasp_candidate"], self.count + 1, 0)
        achieved = self.count >= self.hold_frames
        first_grasp = achieved & ~self.grasped
        self.grasped |= achieved
        first_place = placement_success & self.grasped & ~self.placed
        self.placed |= first_place
        # Approach/contact bonuses cannot be collected indefinitely after lifting.
        approach = (~self.grasped).float()
        self.signals = {name: signals[name] * approach for name in ("reach", "alignment", "aperture", "contact")}
        self.signals["lift"] = signals["lift"]
        self.signals["transport"] = signals["transport"] * self.grasped.float()
        # RewardManager multiplies by dt: divide event indicators here so weights
        # are actual one-time bonuses (+5/+20), independent of the control rate.
        self.signals["grasp_bonus"] = first_grasp.float() / self.dt
        self.signals["placement_bonus"] = first_place.float() / self.dt
        return self.signals

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.count[ids] = 0
        self.grasped[ids] = False
        self.placed[ids] = False
        for value in self.signals.values():
            value[ids] = 0
        # Preserve the step cache for other environments. The next physics step
        # has a new common counter; multiple term reset calls remain idempotent.
