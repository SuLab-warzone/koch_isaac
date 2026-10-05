#!/usr/bin/env python3
"""Start Koch simulation, then dispatch preview, policy evaluation, or PPO training.

Read this file first. Details live in:
  scene_cli.py      -- grouped command-line options and pre-launch checks
  scene_runtime.py  -- environment construction, mode dispatch, and cleanup
  scene_preview.py  -- manual joint targets, gentle motion, and smoke checks

Existing ./run.sh commands and option names remain unchanged.
"""
from pathlib import Path
import sys
import traceback

# Make the local project importable without installing it into either conda env.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scene_cli import build_parser, prepare_run


def main(argv=None):
    # AppLauncher itself can be imported before Kit starts. Scene configs, pxr,
    # sensors and other simulator modules must wait until AFTER AppLauncher().
    from isaaclab.app import AppLauncher

    parser = build_parser()
    AppLauncher.add_app_launcher_args(parser)  # --headless, --device, etc.
    args = parser.parse_args(argv)
    calibration, target_radians = prepare_run(parser, args)

    # Reject invalid options/missing policy files before the expensive startup.
    app = AppLauncher(args).app
    exit_code = 0
    try:
        from scene_runtime import run_scene

        run_scene(args, calibration, target_radians)
    except BaseException:
        # Keep a visible traceback and a failing exit status for CLI automation.
        # The runtime closes its environment; this outer layer owns the Kit app.
        traceback.print_exc()
        exit_code = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        app.close(exit_code=exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
