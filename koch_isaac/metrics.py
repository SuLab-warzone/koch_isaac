"""Vector episode metrics, evaluated before the environment auto-resets."""
import math
import torch


class EpisodeMetrics:
    def __init__(self, num_envs, device, dt, grasp_hold=0.2, place_hold=0.5):
        self.grasp_frames = max(1, math.ceil(grasp_hold / dt))
        self.place_frames = max(1, math.ceil(place_hold / dt))
        self.grasp_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.place_count = torch.zeros_like(self.grasp_count)
        self.grabbed = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.placed = torch.zeros_like(self.grabbed)
        self.peak_lift = torch.zeros(num_envs, device=device)

    def update(self, grasp_candidate, place_candidate, lift_height):
        self.grasp_count = torch.where(grasp_candidate, self.grasp_count + 1, 0)
        self.place_count = torch.where(place_candidate, self.place_count + 1, 0)
        self.grabbed |= self.grasp_count >= self.grasp_frames
        # Placement is a condition at the end, not an ever-achieved latch.
        self.placed = self.place_count >= self.place_frames
        self.peak_lift = torch.maximum(self.peak_lift, lift_height)

    def reset(self, ids):
        for value in (self.grasp_count, self.place_count, self.grabbed, self.placed, self.peak_lift):
            value[ids] = 0

    def outcome(self, index):
        return {"grab_success": bool(self.grabbed[index]),
                "placement_success": bool(self.placed[index]),
                "pick_place_success": bool(self.grabbed[index] & self.placed[index]),
                "peak_lift_m": float(self.peak_lift[index])}
