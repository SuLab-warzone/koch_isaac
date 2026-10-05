"""Force units, simultaneous contact, and vector reset regression checks."""
import json
import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from koch_isaac.metrics import ContactForceMetrics


class ContactForceTests(unittest.TestCase):
    def make_metrics(self, **overrides):
        params = dict(num_envs=2, device="cpu", dt=1/30,
                      contact_threshold=0.01, mass_kg=0.010,
                      gravity_m_s2=9.81, static_friction=0.8, dynamic_friction=0.6)
        params.update(overrides)
        return ContactForceMetrics(**params)

    def test_opposing_contacts_and_lift_reference(self):
        m = self.make_metrics()
        m.update(torch.tensor([[[0.2, 0., 0.], [-0.3, 0., 0.]],
                               [[0., 0., 0.], [0., 0., 0.]]]))
        row = m.outcome(0)
        self.assertAlmostEqual(row["static_finger_contact"]["mean_normal_force_N"], 0.2)
        self.assertAlmostEqual(row["moving_finger_contact"]["peak_normal_force_N"], 0.3)
        self.assertEqual(row["dual_contact_fraction"], 1.0)
        self.assertAlmostEqual(row["longest_dual_contact_s"], 1/30)
        reference = row["lift_reference"]
        self.assertAlmostEqual(reference["minimum_upward_support_N"], 0.0981)
        self.assertAlmostEqual(reference["ideal_normal_per_finger_static_N"], 0.0613125)
        self.assertAlmostEqual(reference["ideal_normal_per_finger_sliding_N"], 0.08175)
        self.assertEqual(m.outcome(1)["dual_contact_fraction"], 0.0)
        json.dumps(row, allow_nan=False)

    def test_separate_hits_are_not_simultaneous_and_final_release_is_retained(self):
        m = self.make_metrics()
        for values in ((1., 0.), (0., 2.), (0.1, 0.2), (0., 0.)):
            force = torch.zeros((2, 2, 3))
            force[0, :, 0] = torch.tensor(values)
            m.update(force)
        row = m.outcome(0)
        self.assertAlmostEqual(row["peak_simultaneous_weaker_finger_normal_N"], 0.1)
        self.assertEqual(row["dual_contact_fraction"], 0.25)
        self.assertEqual(row["static_finger_contact"]["peak_normal_force_N"], 1.0)
        self.assertEqual(row["moving_finger_contact"]["peak_normal_force_N"], 2.0)
        self.assertEqual(row["static_finger_contact"]["final_normal_force_N"], 0.0)
        self.assertAlmostEqual(row["static_finger_contact"]["mean_normal_force_N"], 1.1/4)
        self.assertEqual(row["static_finger_contact"]["contact_fraction"], 0.5)

    def test_selective_reset_preserves_other_episode_and_terminal_snapshot(self):
        m = self.make_metrics()
        m.update(torch.ones((2, 2, 3)))
        terminal = m.outcome(0)
        other = m.outcome(1)
        m.reset(torch.tensor([0]))
        self.assertEqual(m.outcome(0)["sample_count"], 0)
        self.assertEqual(m.outcome(0)["static_finger_contact"]["peak_normal_force_N"], 0.0)
        self.assertEqual(m.outcome(1), other)
        self.assertEqual(terminal["sample_count"], 1)
        self.assertGreater(terminal["static_finger_contact"]["final_normal_force_w_N"][0], 0)

    def test_reference_uses_supplied_properties_and_handles_zero_friction(self):
        m = self.make_metrics(mass_kg=0.02, gravity_m_s2=10., static_friction=0.)
        row = m.outcome(0)
        self.assertAlmostEqual(row["lift_reference"]["minimum_upward_support_N"], 0.2)
        self.assertIsNone(row["lift_reference"]["ideal_normal_per_finger_static_N"])
        json.dumps(row, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
