"""Manual sequence timing, input validation, and launch contracts without Isaac."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from grasp_sequence import load_sequence
from scene_cli import build_parser, prepare_run
import scene_runtime
from utils import sim_to_encoder, JOINT_NAMES, ZERO_COUNTS


def example():
    # Synthetic waypoints for unit tests, not a validated picking trajectory.
    return {"units": "radians", "move_seconds": 2, "hold_seconds": 1,
            "open": [0, 0, 0, 0, 0, 1], "align": [0.1, 0.2, 0, 0, 0, 1],
            "close": [0.1, 0.2, 0, 0, 0, 0.2], "lift": [0.1, 0.3, 0, 0, 0, 0.2]}


class GraspSequenceTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "sequence.json"

    def write(self, data):
        self.path.write_text(json.dumps(data))
        return self.path

    def parse(self, *argv):
        parser = build_parser()
        parser.set_defaults(headless=True, visualizer=None, device=None, enable_cameras=False)
        return parser, parser.parse_args(argv)

    def test_interpolation_holds_boundaries_and_final_pose(self):
        seq = load_sequence(self.write(example()), {})
        initial = np.zeros(6)
        self.assertEqual(seq.duration, 12)
        np.testing.assert_allclose(seq.sample(0, initial)[1], initial)
        np.testing.assert_allclose(seq.sample(1, initial)[1], seq.poses[0] / 2)
        np.testing.assert_allclose(seq.sample(2.5, initial)[1], seq.poses[0])
        self.assertEqual(seq.sample(3, initial)[0], "align")
        np.testing.assert_allclose(seq.sample(3, initial)[1], seq.poses[0])
        np.testing.assert_allclose(seq.sample(7, initial)[1], (seq.poses[1] + seq.poses[2]) / 2)
        for t in (11, 12, 30):
            np.testing.assert_allclose(seq.sample(t, initial)[1], seq.poses[-1])

    def test_encoder_conversion(self):
        data = example()
        expected = np.array([data[p] for p in ("open", "align", "close", "lift")])
        data["units"] = "encoder"
        for phase in ("open", "align", "close", "lift"):
            data[phase] = sim_to_encoder(data[phase]).tolist()
        np.testing.assert_allclose(load_sequence(self.write(data), {}).poses, expected, atol=1e-12)

    def test_lerobot_conversion(self):
        calibration = {n: {"range_min": float(z), "range_max": float(z + 800)}
                       for n, z in zip(JOINT_NAMES, ZERO_COUNTS)}
        data = example()
        data.update(units="lerobot", open=[0]*5+[80], align=[10]*5+[80],
                    close=[10]*5+[20], lift=[20]*5+[20])
        seq = load_sequence(self.write(data), calibration)
        self.assertGreater(seq.poses[0, -1], seq.poses[2, -1])
        self.assertTrue(np.all(seq.poses[1, :5] == seq.poses[2, :5]))

    def test_invalid_and_ambiguous_inputs_rejected(self):
        for key, value in (("open", [None]*6), ("open", [0]*5),
                           ("open", [0]*5+[float("nan")]), ("units", "degrees"),
                           ("move_seconds", 0), ("hold_seconds", float("inf")),
                           ("hold_seconds", True), ("open", [0]*5+[2]),
                           ("close", [0]*6), ("lift", [0]*5+[0.5])):
            with self.subTest(key=key, value=value):
                data = example()
                data[key] = value
                with self.assertRaises(ValueError):
                    load_sequence(self.write(data), {})
        template = ROOT / "config/manual_grasp.template.json"
        with self.assertRaisesRegex(ValueError, "no null placeholders"):
            load_sequence(template, {})

    def test_manual_cli_enables_diagnostics_without_resolving_policy(self):
        parser, args = self.parse("--mode", "grasp-test", "--grasp-sequence", str(self.write(example())))
        with patch("scene_cli.load_calibration", return_value={}), \
             patch("scene_cli._resolve_policy_files") as resolve:
            prepare_run(parser, args)
        resolve.assert_not_called()
        self.assertEqual(args.steps, 360)
        self.assertTrue(args.camera)
        self.assertTrue(args.enable_cameras)

    def test_manual_cli_rejects_incompatible_options(self):
        path = str(self.write(example()))
        for extra in (["--num_envs", "2"], ["--check"], ["--steps", "361"],
                      ["--episode-seconds", "2"], ["--fabric", "off"],
                      ["--radians-target"] + ["0"]*6):
            with self.subTest(extra=extra):
                parser, args = self.parse("--mode", "grasp-test", "--grasp-sequence", path, *extra)
                with patch("scene_cli.load_calibration", return_value={}), \
                     contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    prepare_run(parser, args)

    def test_rerun_allowed_for_manual_mode(self):
        parser, args = self.parse("--mode", "grasp-test", "--grasp-sequence", str(self.write(example())), "--rerun")
        with patch("scene_cli.load_calibration", return_value={}), \
             patch("scene_cli.importlib.util.find_spec", return_value=object()):
            prepare_run(parser, args)

    def test_radians_target_and_limits(self):
        parser, args = self.parse("--radians-target", "0", "0", "0", "0", "0", "1")
        with patch("scene_cli.load_calibration", return_value={}):
            _, target = prepare_run(parser, args)
        np.testing.assert_allclose(target, [0]*5+[1])
        parser, args = self.parse("--radians-target", "0", "0", "0", "0", "0", "2")
        with patch("scene_cli.load_calibration", return_value={}), \
             contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            prepare_run(parser, args)

    def test_manual_runtime_enables_sensors_and_extends_horizon(self):
        parser, args = self.parse("--mode", "grasp-test", "--grasp-sequence", str(self.write(example())))
        with patch("scene_cli.load_calibration", return_value={}):
            prepare_run(parser, args)
        cfg = SimpleNamespace(
            scene=SimpleNamespace(num_envs=0), sim=SimpleNamespace(use_fabric=False),
            enable_front_camera=Mock(), enable_grasp_evaluation=Mock(),
            set_reward_stage=Mock(),
        )
        constructor = Mock(return_value=SimpleNamespace())
        with patch.dict(sys.modules, {
            "gymnasium": SimpleNamespace(make=Mock()),
            "koch_isaac": SimpleNamespace(),
            "koch_isaac.env_cfg": SimpleNamespace(KochPickPlaceEnvCfg=lambda: cfg),
            "koch_isaac.evaluation_env": SimpleNamespace(EvaluatedKochEnv=constructor),
        }):
            env = scene_runtime.create_environment(args)
        self.assertIs(env, constructor.return_value)
        self.assertEqual(cfg.episode_length_s, 13)
        self.assertEqual(cfg.scene.num_envs, 1)
        cfg.enable_front_camera.assert_called_once()
        cfg.enable_grasp_evaluation.assert_called_once()
        cfg.set_reward_stage.assert_called_once_with("grasp")
        self.assertEqual(constructor.call_args.kwargs["grasp_hold"], args.grasp_hold)


if __name__ == "__main__":
    unittest.main()
