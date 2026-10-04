"""Coordinate contract checks; run without Isaac Sim or physical hardware."""
import copy
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from utils import (JOINT_NAMES, ZERO_COUNTS, encoder_to_sim, sim_to_encoder,
                   encoder_to_lerobot, lerobot_to_encoder, sim_to_lerobot,
                   lerobot_to_sim, joint_report, load_calibration)

CALIBRATION = {
    name: {"range_min": low, "range_max": high}
    for name, low, high in zip(JOINT_NAMES,
        [0, 1168, 758, 1354, 0, 1434], [4095, 2706, 2923, 3867, 4095, 2329])
}


class ConversionTests(unittest.TestCase):
    def test_measured_zero_and_encoder_quarter_turn(self):
        np.testing.assert_array_equal(encoder_to_sim(ZERO_COUNTS), np.zeros(6))
        np.testing.assert_allclose(encoder_to_sim(ZERO_COUNTS + 1024), np.pi / 2)
        np.testing.assert_allclose(encoder_to_sim(ZERO_COUNTS - 1024), -np.pi / 2)

    def test_batch_round_trip_and_no_encoder_clipping(self):
        counts = np.array([[2105, 1615, 1768, 2630, 1405, 1447],
                           [-200, 5000, 2748, 3608, 2435, 2300]])
        np.testing.assert_array_equal(sim_to_encoder(encoder_to_sim(counts), round_counts=True), counts)

    def test_lerobot_endpoints_midpoints_and_clipping(self):
        low = np.array([CALIBRATION[n]["range_min"] for n in JOINT_NAMES])
        high = np.array([CALIBRATION[n]["range_max"] for n in JOINT_NAMES])
        expected = [[-100]*5 + [0], [0]*5 + [50], [100]*5 + [100]]
        np.testing.assert_allclose(encoder_to_lerobot([low, (low+high)/2, high], CALIBRATION), expected)
        np.testing.assert_array_equal(lerobot_to_encoder([-200]*6, CALIBRATION), low)
        np.testing.assert_array_equal(lerobot_to_encoder([200]*6, CALIBRATION), high)
        np.testing.assert_array_equal(encoder_to_lerobot(high+100, CALIBRATION), [100]*6)

    def test_bus_truncation_and_degrees_convention(self):
        # Half of [0,4095] is 2047.5: LeRobot truncates rather than rounding.
        self.assertEqual(lerobot_to_encoder([0]*5+[50], CALIBRATION)[0], 2047)
        low = np.array([CALIBRATION[n]["range_min"] for n in JOINT_NAMES])
        high = np.array([CALIBRATION[n]["range_max"] for n in JOINT_NAMES])
        counts = (low+high)/2
        counts[:5] += 4095/4
        np.testing.assert_allclose(encoder_to_lerobot(counts, CALIBRATION, use_degrees=True), [90]*5+[50])
        np.testing.assert_array_equal(lerobot_to_encoder([90]*5+[50], CALIBRATION, use_degrees=True), np.trunc(counts))

    def test_full_policy_boundary_with_tick_quantization(self):
        counts = np.array([2400, 1700, 2200, 2800, 2000, 2100])
        q = encoder_to_sim(counts)
        values = sim_to_lerobot(q, CALIBRATION)
        recovered = lerobot_to_sim(values, CALIBRATION)
        self.assertLessEqual(np.max(np.abs(recovered-q)), 2*np.pi/4096 + 1e-12)
        report = joint_report(encoder_to_sim([0, 0, 0, 0, 0, 0]), CALIBRATION)
        self.assertTrue(report["outside_calibration_range"][1])

    def test_bad_inputs_fail(self):
        for value in ([0]*5, [0]*5+[float("nan")], 0):
            with self.assertRaises(ValueError): encoder_to_sim(value)
        bad = copy.deepcopy(CALIBRATION)
        bad["shoulder_pan"]["range_max"] = 0
        with self.assertRaises(ValueError): encoder_to_lerobot(ZERO_COUNTS, bad)


if __name__ == "__main__":
    unittest.main()
