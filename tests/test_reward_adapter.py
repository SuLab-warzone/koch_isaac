"""Reward adapter buffer/geometry/cache contracts with mocked Isaac interfaces."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from koch_isaac.reward_shaping import GraspShapingSettings


class Scene(dict):
    pass


class FakeManagerTermBase:
    def __init__(self, cfg, env):
        self._env = env


def load_adapter():
    # Quaternion API calls use known rotation matrices here. These tests exercise
    # the adapter's geometry/frames, not Isaac's quaternion implementation.
    package = ModuleType("koch_isaac.mdp")
    package.__path__ = [str(ROOT / "koch_isaac/mdp")]
    observations = SimpleNamespace(
        box_position=lambda env: env.scene["box"].data.root_pos_w.torch - env.scene.env_origins,
        box_com_position=lambda env: env.scene["box"].data.root_com_pos_w.torch - env.scene.env_origins,
        tcp_position=lambda env: env.tcp,
        goal_position=lambda env: env.goal,
    )
    spec = importlib.util.spec_from_file_location("koch_isaac.mdp.rewards", ROOT / "koch_isaac/mdp/rewards.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {
        "koch_isaac.mdp": package,
        "koch_isaac.mdp.observations": observations,
        "koch_isaac.mdp.terminations": SimpleNamespace(placement_candidate=lambda env: env.success),
        "isaaclab.managers": SimpleNamespace(ManagerTermBase=FakeManagerTermBase),
        "isaaclab.utils.math": SimpleNamespace(
            matrix_from_quat=lambda matrices: matrices,
            quat_apply_inverse=lambda matrices, vectors: torch.bmm(matrices.transpose(-1, -2), vectors.unsqueeze(-1)).squeeze(-1),
        ),
    }):
        spec.loader.exec_module(module)
    return module


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.adapter = load_adapter()
        origins = torch.tensor([[0., 0., 0.], [3., 2., 1.]])
        local = torch.tensor([[0.13, 0., 0.044], [0.13, 0., 0.044]])
        wrap = lambda t: SimpleNamespace(torch=t)
        scene = Scene()
        scene.env_origins = origins
        scene["box"] = SimpleNamespace(data=SimpleNamespace(
            root_pos_w=wrap(local + origins), root_com_pos_w=wrap(local + origins),
            root_quat_w=wrap(torch.eye(3).repeat(2, 1, 1)),
        ))
        scene["robot"] = SimpleNamespace(
            joint_names=["follower_joint_gripper"], find_bodies=lambda _: ([0], ["static"]),
            data=SimpleNamespace(body_quat_w=wrap(torch.eye(3).repeat(2, 1, 1, 1)),
                                 joint_pos=wrap(torch.zeros(2, 1))),
        )
        for name, sign in (("static_finger_contact", 1), ("moving_finger_contact", -1)):
            force = torch.tensor([sign * 0.1, 0., 0.]).repeat(2, 1, 1, 1)
            scene[name] = SimpleNamespace(data=SimpleNamespace(force_matrix_w=wrap(force)))
        self.env = SimpleNamespace(
            scene=scene, tcp=local.clone(), goal=local.clone(), num_envs=2,
            device="cpu", step_dt=0.1, common_step_counter=1,
            success=torch.zeros(2, dtype=torch.bool),
        )
        self.env.termination_manager = SimpleNamespace(get_term=lambda _: self.env.success)

    def test_world_offsets_are_removed_and_warp_force_buffers_supported(self):
        signals = self.adapter.grasp_snapshot(self.env, GraspShapingSettings())
        torch.testing.assert_close(signals["reach"], torch.ones(2), atol=2e-5, rtol=0)
        torch.testing.assert_close(signals["lift"], torch.ones(2), atol=2e-5, rtol=0)
        torch.testing.assert_close(signals["contact"], torch.ones(2))

    def test_height_uses_oriented_corners_not_center(self):
        # Roll box 90 degrees: its 20 mm side is vertical, not its 8 mm thickness.
        rotation = torch.tensor([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
        self.env.scene["box"].data.root_quat_w.torch[:] = rotation
        self.env.scene["box"].data.root_pos_w.torch[:, 2] = self.env.scene.env_origins[:, 2] + 0.015
        signals = self.adapter.grasp_snapshot(self.env, GraspShapingSettings())
        torch.testing.assert_close(signals["lift"], torch.full((2,), 0.125), atol=2e-5, rtol=0)

    def test_shared_terms_sample_once_and_do_not_use_poststep_metrics(self):
        reach = self.adapter.GraspReward(None, self.env)
        bonus = self.adapter.GraspReward(None, self.env)
        self.assertIs(reach.state, bonus.state)
        with patch.object(self.adapter, "grasp_snapshot", wraps=self.adapter.grasp_snapshot) as snapshot:
            for _ in range(8):
                reach(self.env, "reach")
                bonus(self.env, "grasp_bonus")
            self.assertEqual(snapshot.call_count, 1)
            self.env.common_step_counter += 1
            torch.testing.assert_close(bonus(self.env, "grasp_bonus"), torch.full((2,), 10.))
            self.assertEqual(snapshot.call_count, 2)
            torch.testing.assert_close(reach(self.env, "reach"), torch.zeros(2))
        bonus.reset(torch.tensor([0]))
        self.assertEqual(bonus.state.grasped.tolist(), [False, True])

    def test_contact_sensor_shape_mismatch_is_explicit(self):
        sensor = self.env.scene["static_finger_contact"]
        sensor.data.force_matrix_w.torch = torch.zeros(2, 1, 2, 3)
        with self.assertRaisesRegex(RuntimeError, "one finger/box pair"):
            self.adapter.grasp_snapshot(self.env, GraspShapingSettings())


if __name__ == "__main__":
    unittest.main()
