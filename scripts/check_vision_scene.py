#!/usr/bin/env python3
"""Opt-in rendered-camera projection check; no ACT checkpoint or training needed."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--diagnostic-dir", type=Path,
                    help="Optionally export env0 RGB and calibration for vision_cli.py")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app
exit_code = 0
try:
    import numpy as np
    import torch
    from koch_isaac.env_cfg import KochPickPlaceEnvCfg
    from isaaclab.envs import ManagerBasedRLEnv
    from koch_isaac import settings as s
    from koch_isaac.vision import load_config
    from koch_isaac.vision.isaac_adapter import IsaacBoxVision

    cfg = KochPickPlaceEnvCfg()
    cfg.scene.num_envs = 2
    cfg.enable_front_camera()
    cfg.sim.use_fabric = True
    if args.device:
        cfg.sim.device = args.device
    env = ManagerBasedRLEnv(cfg)
    try:
        env.reset(seed=42)
        home = torch.tensor(s.HOME, device=env.device).repeat(2, 1)
        vision = IsaacBoxVision(env, load_config())
        for case in range(3):
            # Ground-truth positions are used ONLY to arrange/score this test.
            # The production estimator reads RGB, camera calibration and robot FK.
            positions = torch.tensor([[.08 + case*.02, -.04, .004],
                                      [.09 + case*.02, .01, .004]], device=env.device)
            box = env.scene["box"]
            state = box.data.default_root_state.torch.clone()
            state[:, :3] = positions + env.scene.env_origins
            angle = case * .25
            state[:, 3:7] = torch.tensor([0., 0., np.sin(angle/2), np.cos(angle/2)], device=env.device)
            state[:, 7:] = 0
            box.write_root_state_to_sim(state)
            with torch.inference_mode():
                for _ in range(10):
                    env.step(home)
            rgb = env.scene["front_camera"].data.output["rgb"][..., :3].detach().cpu().numpy()
            vision.observe(rgb, np.ones(2))
            actual = (box.data.root_com_pos_w.torch - env.scene.env_origins).detach().cpu().numpy()
            for i, estimate in enumerate(vision.last_estimates):
                assert estimate.valid, (case, i, estimate.as_dict())
                error = float(np.linalg.norm(estimate.position[:2] - actual[i, :2]))
                print(json.dumps({"case": case, "env": i, **estimate.as_dict(), "xy_error_m": error}), flush=True)
                assert error < .005, "Camera projection error exceeds 5 mm"
            if case == 0 and args.diagnostic_dir:
                from PIL import Image
                args.diagnostic_dir.mkdir(parents=True, exist_ok=True)
                Image.fromarray(rgb[0]).save(args.diagnostic_dir / "front.png")
                (args.diagnostic_dir / "calibration.json").write_text(
                    json.dumps(vision.estimators[0].calibration.as_dict(), indent=2))
        print("PASS: RGB-only projection works at three box locations in two environment origins", flush=True)
    finally:
        env.close()
except BaseException:
    import traceback
    traceback.print_exc()
    exit_code = 1
finally:
    app.close(exit_code=exit_code)
