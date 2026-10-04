#!/usr/bin/env python3
"""Preview, gently exercise joints, or run a finite headless smoke check."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import (DEFAULT_CALIBRATION, SIM_JOINT_NAMES, encoder_to_sim,
                   joint_report, lerobot_to_sim, load_calibration)
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=0, help="0: run until window closes; headless default 300")
parser.add_argument("--mode", choices=("hold", "joints", "act"), default="hold")
parser.add_argument("--camera", action="store_true")
parser.add_argument("--fabric", choices=("auto", "on", "off"), default="auto",
                    help="auto: Fabric enabled for current rendered poses; off is diagnostic only")
parser.add_argument("--snapshot", type=Path, help="Save front RGB image; implies --camera")
parser.add_argument("--check", action="store_true", help="Check state, contacts, reset isolation and success metric")
targets = parser.add_mutually_exclusive_group()
targets.add_argument("--encoder-target", type=float, nargs=6, metavar="COUNT",
                     help="Hold six reported encoder counts, in shoulder pan/lift, elbow, wrist flex/roll, gripper order")
targets.add_argument("--lerobot-target", type=float, nargs=6, metavar="VALUE",
                     help="Hold six postprocessed LeRobot motor values in the same order")
parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
parser.add_argument("--lerobot-use-degrees", action="store_true",
                    help="Use only if the training robot used use_degrees=True; gripper remains [0,100]")
parser.add_argument("--print-joints-every", type=int, default=0,
                    help="Print converted measured joints every N steps; 0: final report only")
parser.add_argument("--episodes", type=int, default=5, help="ACT: number of completed episodes")
parser.add_argument("--policy-path", type=Path, help="ACT: local pretrained directory; defaults to cached home_v2")
parser.add_argument("--policy-python", type=Path,
                    default=Path("/home/niel/miniforge3/envs/lerobot061/bin/python"))
parser.add_argument("--policy-device", default="cuda", help="ACT worker device; simulator uses --device")
parser.add_argument("--output-dir", type=Path, help="ACT: results directory; default outputs/act_TIMESTAMP")
parser.add_argument("--video", action="store_true", help="ACT: record front.mp4 at 30 simulated fps")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--log-every", type=int, default=30, help="ACT: action log interval in steps")
parser.add_argument("--episode-seconds", type=float, help="ACT: override the default 20-second episode")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.steps < 0: parser.error("--steps must be nonnegative")
if args.num_envs < 1: parser.error("--num_envs must be positive")
if args.print_joints_every < 0: parser.error("--print-joints-every must be nonnegative")
if args.mode != "hold" and (args.encoder_target is not None or args.lerobot_target is not None):
    parser.error("Explicit joint targets require --mode hold")
try:
    calibration = load_calibration(args.calibration)
    target_radians = None
    if args.encoder_target is not None:
        target_radians = encoder_to_sim(args.encoder_target)
    elif args.lerobot_target is not None:
        target_radians = lerobot_to_sim(args.lerobot_target, calibration, use_degrees=args.lerobot_use_degrees)
except (OSError, ValueError, KeyError, TypeError) as error:
    parser.error(str(error))
if args.mode == "act":
    if args.num_envs != 1: parser.error("ACT rollout requires --num_envs 1")
    if args.check: parser.error("Use --check separately from ACT evaluation")
    if args.fabric == "off": parser.error("ACT requires Fabric for current camera poses; use --fabric on")
    if args.episodes < 1 or args.log_every < 1: parser.error("--episodes and --log-every must be positive")
    if args.episode_seconds is not None and (not 0 < args.episode_seconds < float("inf")):
        parser.error("--episode-seconds must be finite and positive")
    if not args.policy_python.is_file(): parser.error("LeRobot interpreter not found; set --policy-python")
    if args.policy_path is None:
        baseline = json.loads((Path(__file__).resolve().parents[1] / "config/baseline.json").read_text())
        args.policy_path = (Path.home() / ".cache/huggingface/hub" /
                            ("models--" + baseline["repo_id"].replace("/", "--")) /
                            "snapshots" / baseline["cached_revision"])
    args.policy_path = args.policy_path.expanduser().resolve()
    for filename in ("config.json", "model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json"):
        if not (args.policy_path / filename).is_file():
            parser.error(f"Missing {filename} in checkpoint {args.policy_path}")
    args.camera = True
elif args.video:
    parser.error("--video is supported in --mode act")
if args.snapshot: args.camera = True
if not args.headless and not args.visualizer: args.visualizer = ["kit"]
if args.camera: args.enable_cameras = True
# Start Kit before importing pxr-dependent configs (required by installed Sim 6.0.1).
app = AppLauncher(args).app
exit_code = 0
try:
    import torch
    from koch_isaac.env_cfg import KochPickPlaceEnvCfg
    from koch_isaac import settings as s
    cfg = KochPickPlaceEnvCfg()
    cfg.scene.num_envs = args.num_envs
    if args.mode == "act":
        cfg.num_rerenders_on_reset = 2
        if args.episode_seconds is not None: cfg.episode_length_s = args.episode_seconds
    cfg.sim.use_fabric = args.fabric != "off"
    print(f"Scene synchronization: {'Fabric' if cfg.sim.use_fabric else 'USD'}", flush=True)
    if args.device: cfg.sim.device = args.device
    if args.camera: cfg.enable_front_camera()
    import gymnasium as gym
    from PIL import Image
    env = gym.make("Koch-PinkBox-Place-v0", cfg=cfg).unwrapped
    try:
        env.reset(seed=args.seed)
        if args.mode == "act":
            from act_rollout import rollout
            rollout(env, args, calibration)
        else:
            if tuple(s.JOINT_NAMES) != SIM_JOINT_NAMES:
                raise ValueError("Action joint order differs from the calibration mapping")
            robot = env.scene["robot"]
            joint_ids = [robot.joint_names.index(name) for name in SIM_JOINT_NAMES]
            def measured_joints():
                return joint_report(robot.data.joint_pos.torch[0, joint_ids].detach().cpu().numpy(),
                                    calibration, use_degrees=args.lerobot_use_degrees)
            home = torch.tensor(s.HOME, device=env.device).repeat(env.num_envs, 1)
            hold = home if target_radians is None else torch.tensor(
                target_radians, dtype=home.dtype, device=env.device).repeat(env.num_envs, 1)
            print("Joint target: " + json.dumps(joint_report(hold[0].cpu().numpy(), calibration,
                  use_degrees=args.lerobot_use_degrees)), flush=True)
            print("Targets pass through the environment's configured joint limits; calibration is provisional.", flush=True)
            limit = args.steps or (300 if args.headless else 10**9)
            for step in range(limit):
                if env.sim.visualizers and not any(v.is_running() and not v.is_closed for v in env.sim.visualizers): break
                action = hold.clone()
                if args.mode == "joints":
                    # Small motion around home, one joint at a time. This is NOT a pick policy.
                    joint = (step//90)%6
                    action[:, joint] += 0.15*torch.sin(torch.tensor((step%90)/90*2*torch.pi, device=env.device))
                with torch.inference_mode():
                    obs, reward, terminated, truncated, extras = env.step(action)
                if args.print_joints_every and (step + 1) % args.print_joints_every == 0:
                    print("Joint state: " + json.dumps(measured_joints()), flush=True)
                if args.check:
                    assert torch.isfinite(obs["policy"]).all(), "Non-finite observation"
                    assert torch.isfinite(reward).all(), "Non-finite reward"
            if args.snapshot:
                for _ in range(5): env.sim.render()
                rgb = env.scene["front_camera"].data.output["rgb"][0, ..., :3].cpu().numpy()
                args.snapshot.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(rgb).save(args.snapshot)
                print(f"Saved camera: {args.snapshot}")
            if args.check:
                from smoke_checks import check_scene
                check_scene(env, home)
            print(json.dumps({"steps": step+1, "num_envs": env.num_envs,
                "joint_names": env.scene["robot"].joint_names,
                "joint_positions": env.scene["robot"].data.joint_pos.torch[0].tolist(),
                "joint_conversion": measured_joints(),
                "box_local": (env.scene["box"].data.root_pos_w.torch-env.scene.env_origins)[0].tolist(),
                "observation_shape": list(obs["policy"].shape),
                "use_fabric": cfg.sim.use_fabric, "device": str(env.device), "status": "PASS"}, indent=2))
    finally:
        env.close()

except BaseException:
    import traceback
    traceback.print_exc()
    exit_code = 1
finally:
    sys.stdout.flush()
    sys.stderr.flush()
    app.close(exit_code=exit_code)
