"""Tests for actual perception geometry, missing detections, and PPO input isolation."""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT)]
from koch_isaac.vision import ColorPlaneEstimator, PlaneCalibration, load_config
from koch_isaac.vision.base import BoxEstimate
from koch_isaac.vision.features import PlanarBoxFeatures
from residual_observations import ACTOR_DIM, CRITIC_DIM, make_residual_observation, terminal_residual_observation
from residual_ppo import ResidualPolicy, load_residual


class VisionTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()
        self.config["workspace_xy_m"] = [[-1, 1], [-1, 1]]
        self.calibration = PlaneCalibration(np.array([[.001, 0, -.1], [0, .001, -.1], [0, 0, 1]]),
                                            (200, 200), .008, .004)
        self.estimator = ColorPlaneEstimator(self.calibration, self.config)

    def image(self):
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        image[90:110, 110:130] = [240, 40, 140]  # RGB pink, centroid (119.5,99.5)
        return image

    def test_rgb_detection_and_missing_do_not_return_stale_position(self):
        result = self.estimator.estimate(self.image())
        self.assertTrue(result.valid)
        np.testing.assert_allclose(result.position, [.0195, -.0005, .004], atol=1e-7)
        missing = self.estimator.estimate(np.zeros((200, 200, 3), dtype=np.uint8))
        self.assertFalse(missing.valid)
        np.testing.assert_equal(missing.position, 0)

    def test_similar_pink_objects_are_ambiguous(self):
        image = self.image()
        image[30:50, 30:50] = [240, 40, 140]
        self.assertEqual(self.estimator.estimate(image).reason, "ambiguous_components")

    def test_workspace_and_resolution_reject_wrong_coordinates(self):
        self.estimator.config = {**self.config, "workspace_xy_m": [[.1, .2], [.1, .2]]}
        self.assertFalse(self.estimator.estimate(self.image()).valid)
        with self.assertRaises(ValueError):
            self.estimator.estimate(self.image()[:100])

    def test_projection_uses_camera_axes_and_plane_height(self):
        # Camera above table, optical Z down, X right, Y toward environment -Y.
        k = np.array([[100., 0, 100], [0, 100., 100], [0, 0, 1]])
        r = np.diag([1., -1., -1.])
        calibration = PlaneCalibration.from_camera(k, r, [.1, .2, 1.008], (200, 200), .008, .004)
        np.testing.assert_allclose(calibration.project([110, 120]), [.2, 0., .004], atol=1e-6)

    def test_measured_point_calibration_and_degenerate_points(self):
        pixels = [[0, 0], [200, 0], [200, 200], [0, 200]]
        xy = [[-.1, -.1], [.1, -.1], [.1, .1], [-.1, .1]]
        calibration = PlaneCalibration.from_points(pixels, xy, (200, 200), .008, .004)
        np.testing.assert_allclose(calibration.project([150, 50]), [.05, -.05, .004], atol=1e-6)
        with self.assertRaises(ValueError):
            PlaneCalibration.from_points([[0, 0], [1, 1], [2, 2], [3, 3]], xy, (200, 200), 0, 0)

    def test_grasp_gate_and_reset_are_per_environment(self):
        features = PlanarBoxFeatures(2, self.config)
        estimates = [BoxEstimate(np.array([.1, 0., .004]), True, "test") for _ in range(2)]
        tcp = np.array([[.1, 0., .03], [.1, 0., .03]])
        np.testing.assert_equal(features.update(estimates, tcp, [1., 1.])[:, 3], [1, 1])
        features.note_gripper([.3, 1.])
        np.testing.assert_equal(features.update(estimates, tcp, [.3, 1.])[:, 3], [0, 1])
        # A later reappearance or reopening must not resurrect a possibly lifted cue.
        np.testing.assert_equal(features.update(estimates, tcp, [1., 1.])[0], 0)
        features.reset([0])
        np.testing.assert_equal(features.update(estimates, tcp, [1., 1.])[:, 3], [1, 1])
        missing = [BoxEstimate.missing("hidden"), estimates[1]]
        np.testing.assert_equal(features.update(missing, tcp, [1., 1.])[0], 0)

    def test_vision_actor_does_not_gain_privileged_inputs(self):
        z = np.zeros((2, 6), dtype=np.float32)
        vision = np.array([[.1, .2, .3, 1], [0, 0, 0, 0]], dtype=np.float32)
        privileged = np.zeros((2, CRITIC_DIM), dtype=np.float32)
        observation = make_residual_observation(z, z, z, z, np.zeros(2), privileged, vision)
        np.testing.assert_equal(observation[:, 25:29], vision)
        policy = ResidualPolicy(spaces.Box(-np.inf, np.inf, (60,), np.float32),
                                spaces.Box(-1., 1., (6,), np.float32), lambda _: 3e-4)
        x = torch.as_tensor(observation)
        changed_critic = x.clone()
        changed_critic[:, 29:] = 99
        changed_actor = x.clone()
        changed_actor[:, :29] = 99
        with torch.no_grad():
            torch.testing.assert_close(policy.mlp_extractor.forward_actor(x),
                                       policy.mlp_extractor.forward_actor(changed_critic))
            torch.testing.assert_close(policy.predict_values(x), policy.predict_values(changed_actor))
        terminal = terminal_residual_observation(np.arange(CRITIC_DIM), 29)
        np.testing.assert_equal(terminal[29:], np.arange(CRITIC_DIM))

    def test_new_and_legacy_checkpoints_load_and_wrong_contract_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            for actor_dim in (25, 29):
                # A lightweight real PPO instance verifies SB3 serialization of the layout.
                from stable_baselines3.common.env_util import make_vec_env
                import gymnasium as gym
                class DummyEnv(gym.Env):
                    observation_space = spaces.Box(-1., 1., (actor_dim + CRITIC_DIM,), np.float32)
                    action_space = spaces.Box(-1., 1., (6,), np.float32)
                    def reset(self, *, seed=None, options=None):
                        super().reset(seed=seed)
                        return np.zeros(self.observation_space.shape, np.float32), {}
                env = make_vec_env(DummyEnv)
                model = PPO(ResidualPolicy, env, n_steps=2, batch_size=2, device="cpu")
                model.residual_contract = {"actor_dim": actor_dim}
                path = Path(folder) / str(actor_dim)
                model.save(path)
                loaded = load_residual(path, model.residual_contract)
                self.assertEqual(loaded.policy.mlp_extractor.actor_dim, actor_dim)
                with self.assertRaises(ValueError):
                    load_residual(path, {"actor_dim": 999})
                env.close()


if __name__ == "__main__":
    unittest.main()
