"""Actual tensor reward calculations and milestone/reset invariants without Kit."""
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from koch_isaac.reward_shaping import GraspShapingSettings, GraspRewardState, dense_signals


class DenseRewardTests(unittest.TestCase):
    def signals(self, count=2, **overrides):
        inputs = dict(
            local_error=torch.zeros(count, 3), orientation_score=torch.ones(count),
            gripper_angle=torch.zeros(count),
            finger_forces=torch.tensor([[[0.1, 0., 0.], [-0.1, 0., 0.]]] * count),
            bottom_height=torch.full((count,), 0.04), goal_distance=torch.zeros(count),
            settings=GraspShapingSettings(),
        )
        inputs.update(overrides)
        return dense_signals(**inputs)

    def test_gradual_lift_and_height_saturation(self):
        result = self.signals(count=5, bottom_height=torch.tensor([-0.001, 0., 0.005, 0.04, 0.10]))
        torch.testing.assert_close(result["lift"], torch.tensor([0., 0., 0.125, 1., 1.]))
        self.assertEqual(result["grasp_candidate"].tolist(), [False, False, False, True, True])

    def test_airborne_or_single_finger_contact_cannot_earn_lift_or_transport(self):
        forces = torch.tensor([[[0., 0., 0.], [0., 0., 0.]],
                               [[0.1, 0., 0.], [0., 0., 0.]]])
        result = self.signals(finger_forces=forces)
        for name in ("contact", "lift", "transport"):
            torch.testing.assert_close(result[name], torch.zeros(2))
        self.assertFalse(result["grasp_candidate"].any())

    def test_nonopposing_contacts_and_remote_box_are_rejected(self):
        forces = torch.tensor([[[0.1, 0., 0.], [0.1, 0., 0.]],
                               [[0.1, 0., 0.], [-0.1, 0., 0.]]])
        error = torch.tensor([[0., 0., 0.], [0.1, 0., 0.]])
        result = self.signals(finger_forces=forces, local_error=error)
        torch.testing.assert_close(result["lift"], torch.zeros(2))
        torch.testing.assert_close(result["contact"], torch.zeros(2))

    def test_force_reward_caps_and_uses_weaker_finger(self):
        forces = torch.tensor([[[0.04, 0., 0.], [-2., 0., 0.]],
                               [[100., 0., 0.], [-100., 0., 0.]]])
        result = self.signals(finger_forces=forces)
        torch.testing.assert_close(result["contact"], torch.tensor([0.5, 1.]))

    def test_alignment_sensitive_to_height_and_orientation(self):
        error = torch.tensor([[0., 0., 0.], [0., 0., 0.01]])
        result = self.signals(local_error=error)
        self.assertEqual(float(result["alignment"][0]), 1)
        self.assertLess(float(result["alignment"][1]), 0.05)
        result = self.signals(orientation_score=torch.tensor([0., 1.]))
        torch.testing.assert_close(result["alignment"], torch.tensor([0., 1.]))

    def test_aperture_wants_open_on_approach_and_closed_only_when_aligned(self):
        forces = torch.zeros(2, 2, 3)
        error = torch.tensor([[0.03, 0., 0.], [0., 0., 0.]])
        open_result = self.signals(local_error=error, finger_forces=forces, gripper_angle=torch.ones(2))
        closed_result = self.signals(local_error=error, finger_forces=forces, gripper_angle=torch.zeros(2))
        self.assertGreater(open_result["aperture"][0], closed_result["aperture"][0])
        self.assertGreater(closed_result["aperture"][1], open_result["aperture"][1])
        # Once in physical contact, the angle objective stops encouraging squeeze.
        held = self.signals(gripper_angle=torch.tensor([0.3, 0.7]))
        torch.testing.assert_close(held["aperture"], torch.ones(2))

    def test_all_dense_signals_bounded_and_finite_with_zero_force(self):
        result = self.signals(finger_forces=torch.zeros(2, 2, 3))
        for name, values in result.items():
            self.assertTrue(torch.isfinite(values).all(), name)
            self.assertTrue(((values >= 0) & (values <= 1)).all(), name)


class MilestoneTests(unittest.TestCase):
    def inputs(self, candidate=(True, True)):
        return {**{k: torch.ones(2) for k in ("reach", "alignment", "aperture", "contact", "lift", "transport")},
                "grasp_candidate": torch.tensor(candidate)}

    def test_once_per_episode_and_once_per_step(self):
        state = GraspRewardState(2, "cpu", 0.1, GraspShapingSettings())
        for _ in range(8):
            result = state.update(1, self.inputs(), torch.zeros(2, dtype=torch.bool))
            torch.testing.assert_close(result["grasp_bonus"], torch.zeros(2))
        self.assertEqual(state.count.tolist(), [1, 1])
        result = state.update(2, self.inputs(), torch.zeros(2, dtype=torch.bool))
        torch.testing.assert_close(result["grasp_bonus"] * 5 * 0.1, torch.full((2,), 5.))
        for name in ("reach", "alignment", "aperture", "contact"):
            torch.testing.assert_close(result[name], torch.zeros(2))
        result = state.update(3, self.inputs(), torch.zeros(2, dtype=torch.bool))
        torch.testing.assert_close(result["grasp_bonus"], torch.zeros(2))
        result = state.update(4, self.inputs(), torch.ones(2, dtype=torch.bool))
        torch.testing.assert_close(result["placement_bonus"] * 20 * 0.1, torch.full((2,), 20.))
        result = state.update(5, self.inputs(), torch.ones(2, dtype=torch.bool))
        torch.testing.assert_close(result["placement_bonus"], torch.zeros(2))

    def test_contact_break_restarts_hold_and_placement_requires_grasp(self):
        state = GraspRewardState(2, "cpu", 0.1, GraspShapingSettings())
        no = torch.zeros(2, dtype=torch.bool)
        state.update(1, self.inputs(), no)
        state.update(2, self.inputs((False, False)), torch.ones(2, dtype=torch.bool))
        self.assertFalse(state.placed.any())
        result = state.update(3, self.inputs(), no)
        self.assertFalse(result["grasp_bonus"].any())
        result = state.update(4, self.inputs(), no)
        self.assertTrue((result["grasp_bonus"] > 0).all())
        result = state.update(5, self.inputs((False, False)), no)
        self.assertFalse(result["grasp_bonus"].any())
        self.assertTrue(state.grasped.all())

    def test_selective_and_repeated_resets_preserve_other_environment(self):
        state = GraspRewardState(2, "cpu", 0.1, GraspShapingSettings())
        yes = torch.ones(2, dtype=torch.bool)
        state.update(1, self.inputs(), yes)
        state.update(2, self.inputs(), yes)
        for _ in range(8):
            state.reset(torch.tensor([0]))
        self.assertEqual(state.grasped.tolist(), [False, True])
        self.assertEqual(state.placed.tolist(), [False, True])
        self.assertEqual(state.count.tolist(), [0, 2])
        state.update(3, self.inputs(), yes)
        result = state.update(4, self.inputs(), yes)
        torch.testing.assert_close(result["grasp_bonus"], torch.tensor([10., 0.]))
        torch.testing.assert_close(result["placement_bonus"], torch.tensor([10., 0.]))
        state.reset()
        self.assertFalse(state.grasped.any())
        self.assertFalse(state.placed.any())

    def test_bonus_magnitude_independent_of_control_rate(self):
        for dt in (1/30, 1/60, 0.01):
            with self.subTest(dt=dt):
                state = GraspRewardState(2, "cpu", dt, GraspShapingSettings())
                total = torch.zeros(2)
                for step in range(1, state.hold_frames + 5):
                    total += state.update(step, self.inputs(), torch.zeros(2, dtype=torch.bool))["grasp_bonus"] * 5 * dt
                torch.testing.assert_close(total, torch.full((2,), 5.))


if __name__ == "__main__":
    unittest.main()
