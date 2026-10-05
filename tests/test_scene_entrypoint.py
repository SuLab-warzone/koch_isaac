"""CLI and lifecycle regression tests without starting Isaac or loading a policy."""
import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from scene_cli import build_parser, prepare_run
import scene_runtime


class SceneCliTests(unittest.TestCase):
    def parse(self, *argv):
        parser = build_parser()
        # These fields normally come from AppLauncher, which tests must not start.
        parser.set_defaults(headless=True, visualizer=None, device=None, enable_cameras=False)
        return parser, parser.parse_args(argv)

    def prepare(self, parser, args):
        with patch("scene_cli.load_calibration", return_value={}):
            return prepare_run(parser, args)

    def test_invalid_mode_combinations_fail_before_calibration_access(self):
        for argv in (
            ["--train"],
            ["--mode", "policy", "--checkpoint", "old.zip"],
            ["--mode", "residual-ppo"],
            ["--mode", "joints", "--encoder-target", "0", "0", "0", "0", "0", "0"],
            ["--mode", "residual-ppo", "--train", "--steps", "10"],
            ["--vision", "color-plane"],
            ["--num_envs", "0"],
        ):
            with self.subTest(argv=argv):
                parser, args = self.parse(*argv)
                with patch("scene_cli.load_calibration") as load, contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        prepare_run(parser, args)
                    self.assertEqual(error.exception.code, 2)
                    load.assert_not_called()

    def test_multi_env_suppresses_recordings_but_keeps_checkpoint(self):
        parser, args = self.parse("--num_envs", "2", "--video", "--snapshot", "frame.png",
                                  "--checkpoint-out", "new.zip")
        self.prepare(parser, args)
        self.assertFalse(args.save_output)
        self.assertFalse(args.video)
        self.assertIsNone(args.snapshot)
        self.assertEqual(args.checkpoint_out, Path("new.zip"))

    def test_encoder_target_is_converted_before_launch(self):
        parser, args = self.parse("--encoder-target", "2105", "1615", "1768", "2630", "1405", "1447")
        _, target = self.prepare(parser, args)
        np.testing.assert_allclose(target, np.zeros(6))

    def test_snapshot_enables_camera(self):
        parser, args = self.parse("--snapshot", "frame.png")
        self.prepare(parser, args)
        self.assertTrue(args.camera)
        self.assertTrue(args.enable_cameras)

    def test_policy_checks_local_files_and_enables_camera(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            parser, args = self.parse("--mode", "policy", "--policy-path", str(path),
                                      "--policy-python", sys.executable)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.prepare(parser, args)
            for name in ("config.json", "model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json"):
                (path / name).touch()
            self.prepare(parser, args)
            self.assertTrue(args.camera)
            self.assertTrue(args.enable_cameras)
            self.assertEqual(args.policy_path, path.resolve())

    def test_entrypoint_import_does_not_launch_or_import_isaac(self):
        code = (
            "import sys; sys.path.insert(0, " + repr(str(ROOT / "scripts")) + "); "
            "import run_scene; assert not any(n.startswith('isaaclab') for n in sys.modules)"
        )
        subprocess.run([sys.executable, "-c", code], check=True)


class SceneLifecycleTests(unittest.TestCase):
    def test_mode_error_still_closes_environment(self):
        env = Mock()
        args = SimpleNamespace(rerun=False)
        with patch.object(scene_runtime, "create_environment", return_value=env), \
             patch.object(scene_runtime, "execute_mode", side_effect=RuntimeError("rollout failed")):
            with self.assertRaisesRegex(RuntimeError, "rollout failed"):
                scene_runtime.run_scene(args, {}, None)
        env.close.assert_called_once()

    def test_logger_close_failure_still_closes_environment(self):
        env, logger = Mock(), Mock()
        logger.close.side_effect = RuntimeError("logger failed")
        logger_module = SimpleNamespace(RerunLogger=Mock(return_value=logger))
        args = SimpleNamespace(rerun=True, rerun_env=0)
        with patch.object(scene_runtime, "create_environment", return_value=env), \
             patch.object(scene_runtime, "execute_mode"), \
             patch.dict(sys.modules, {"koch_isaac.rerun_logger": logger_module}):
            with self.assertRaisesRegex(RuntimeError, "logger failed"):
                scene_runtime.run_scene(args, {}, None)
        env.close.assert_called_once()

    def test_entrypoint_closes_app_with_failure_status(self):
        import run_scene

        app = Mock()
        launcher = Mock(return_value=SimpleNamespace(app=app))
        launcher.add_app_launcher_args = lambda parser: parser.set_defaults(
            headless=True, visualizer=None, device=None, enable_cameras=False
        )
        runtime = Mock()
        runtime.run_scene.side_effect = RuntimeError("simulation failed")
        with patch.dict(sys.modules, {
            "isaaclab.app": SimpleNamespace(AppLauncher=launcher),
            "scene_runtime": runtime,
        }), patch("scene_cli.load_calibration", return_value={}), \
             contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(run_scene.main(["--steps", "1"]), 1)
        launcher.assert_called_once()
        runtime.run_scene.assert_called_once()
        app.close.assert_called_once_with(exit_code=1)


if __name__ == "__main__":
    unittest.main()
