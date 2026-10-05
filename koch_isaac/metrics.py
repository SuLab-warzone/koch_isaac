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


class ContactForceMetrics:
    """Per-episode diagnostics from two box-filtered finger contact sensors.

    Input is [environment, finger, xyz], in world coordinates and newtons.
    These are resultant NORMAL contact forces on each finger, not friction,
    actuator torque, or the total upward force on the box. Sample once per
    control step; peaks can miss contacts between those samples.
    """

    FINGER_NAMES = ("static_finger_contact", "moving_finger_contact")

    def __init__(self, num_envs, device, dt, contact_threshold, mass_kg,
                 gravity_m_s2, static_friction, dynamic_friction):
        self.dt = dt
        self.contact_threshold = contact_threshold
        self.mass_kg = mass_kg
        self.gravity_m_s2 = gravity_m_s2
        self.static_friction = static_friction
        self.dynamic_friction = dynamic_friction
        self.samples = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.last_force = torch.zeros((num_envs, 2, 3), device=device)
        self.force_sum = torch.zeros((num_envs, 2), device=device)
        self.force_peak = torch.zeros_like(self.force_sum)
        self.contact_samples = torch.zeros_like(self.force_sum)
        self.dual_samples = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.dual_streak = torch.zeros_like(self.dual_samples)
        self.dual_streak_peak = torch.zeros_like(self.dual_samples)
        self.weaker_finger_peak = torch.zeros(num_envs, device=device)

    def update(self, normal_forces_w):
        if normal_forces_w.shape != self.last_force.shape:
            raise ValueError("Expected normal contact forces [num_envs, 2, 3]")
        # Keep force magnitudes separate: opposing jaw vectors would cancel if
        # summed together, even though both fingers are squeezing firmly.
        magnitude = torch.linalg.vector_norm(normal_forces_w, dim=-1)
        contact = magnitude > self.contact_threshold
        dual = contact.all(dim=-1)
        self.samples += 1
        self.last_force.copy_(normal_forces_w)
        self.force_sum += magnitude
        self.force_peak = torch.maximum(self.force_peak, magnitude)
        self.contact_samples += contact
        self.dual_samples += dual
        self.dual_streak = torch.where(dual, self.dual_streak + 1, 0)
        self.dual_streak_peak = torch.maximum(self.dual_streak_peak, self.dual_streak)
        # max_t(min(N_static(t), N_moving(t))) uses simultaneous samples.
        # min(max_t(N_static), max_t(N_moving)) could falsely combine separate hits.
        self.weaker_finger_peak = torch.maximum(self.weaker_finger_peak, magnitude.amin(dim=-1))

    def reset(self, ids):
        # Clear only completed environments; other vector episodes keep their history.
        for value in (self.samples, self.last_force, self.force_sum, self.force_peak,
                      self.contact_samples, self.dual_samples, self.dual_streak,
                      self.dual_streak_peak, self.weaker_finger_peak):
            value[ids] = 0

    def outcome(self, index):
        count = int(self.samples[index])
        denominator = max(count, 1)
        weight = self.mass_kg * self.gravity_m_s2
        def required_per_finger(mu):
            # Ideal balanced side pinch: 2 * mu * N >= m * g.
            # Zero friction has no finite solution; JSON null avoids Infinity.
            return weight / (2 * mu) if mu > 0 else None

        fingers = {}
        for finger, name in enumerate(self.FINGER_NAMES):
            final = self.last_force[index, finger]
            fingers[name] = {
                "final_normal_force_w_N": final.tolist(),
                "final_normal_force_N": float(torch.linalg.vector_norm(final)),
                "mean_normal_force_N": float(self.force_sum[index, finger]) / denominator,
                "peak_normal_force_N": float(self.force_peak[index, finger]),
                "contact_fraction": float(self.contact_samples[index, finger]) / denominator,
            }
        return {
            "sample_count": count,
            "sample_period_s": self.dt,
            "contact_threshold_N": self.contact_threshold,
            **fingers,
            "dual_contact_fraction": float(self.dual_samples[index]) / denominator,
            "longest_dual_contact_s": float(self.dual_streak_peak[index]) * self.dt,
            "peak_simultaneous_weaker_finger_normal_N": float(self.weaker_finger_peak[index]),
            "lift_reference": {
                "configured_box_mass_kg": self.mass_kg,
                "gravity_m_s2": self.gravity_m_s2,
                "minimum_upward_support_N": weight,
                "box_static_friction": self.static_friction,
                "box_dynamic_friction": self.dynamic_friction,
                "ideal_normal_per_finger_static_N": required_per_finger(self.static_friction),
                "ideal_normal_per_finger_sliding_N": required_per_finger(self.dynamic_friction),
                "assumptions": "Configured mass and box friction only; equal opposing side contacts, "
                    "vertical gravity, zero acceleration, no safety factor. Actual pair friction also "
                    "depends on finger material and combine mode. Not a measured lift force or success test.",
            },
        }
