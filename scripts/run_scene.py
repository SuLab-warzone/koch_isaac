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
parser.add_argument("--mode", choices=("hold", "joints", "act", "policy", "residual-ppo"), default="hold")
parser.add_argument("--camera", action="store_true")
parser.add_argument("--rerun", action="store_true",
                    help="Open live camera/contact/gripper plots in Rerun (policy modes only; no files saved)")
parser.add_argument("--rerun-env", type=int, default=0,
                    help="Environment index shown by --rerun (default: 0)")
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
parser.add_argument("--episodes", type=int, default=5, help="Completed evaluation episodes per environment")
parser.add_argument("--policy-path", type=Path, help="Local ACT/diffusion pretrained directory; defaults to cached home_v2")
parser.add_argument("--policy-python", type=Path,
                    default=Path("/home/niel/miniforge3/envs/lerobot061/bin/python"))
parser.add_argument("--policy-device", default="cuda", help="Shared ACT/diffusion and PPO device (default: cuda); simulator uses --device")
parser.add_argument("--output-dir", type=Path, help="ACT: results directory; default outputs/act_TIMESTAMP")
parser.add_argument("--video", action="store_true", help="ACT: record front.mp4 at 30 simulated fps")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--log-every", type=int, default=30, help="ACT: action log interval in steps")
parser.add_argument("--episode-seconds", type=float, help="ACT: override the default 20-second episode")
parser.add_argument('--save-output', action=argparse.BooleanOptionalAction, default=True,
                    help='Save rollout images/logs/video; always disabled when num_envs > 1')
parser.add_argument('--policy-batch-size', type=int, default=4, help='Frozen policy inference microbatch (1..8)')
parser.add_argument('--train', action='store_true', help='Train residual PPO instead of evaluating')
parser.add_argument('--checkpoint', type=Path, help='Residual PPO .zip to evaluate or resume')
parser.add_argument('--checkpoint-out', type=Path, help='New trained residual .zip; saved even with --no-save-output')
parser.add_argument('--total-timesteps', type=int, default=100000, help='Training transitions across all environments')
parser.add_argument('--ppo-steps', type=int, default=64, help='Steps per environment in each PPO rollout')
parser.add_argument('--train-log-every', type=int, default=30,
                    help='Extra TensorBoard/console updates every N control steps per env; 0: PPO rollout updates only')
parser.add_argument('--ppo-batch-size', type=int, default=64)
parser.add_argument('--ppo-epochs', type=int, default=4)
parser.add_argument('--learning-rate', type=float, default=3e-4)
parser.add_argument('--residual-limit', type=float, default=0.25, help='Maximum absolute correction per joint in radians')
parser.add_argument('--residual-penalty', type=float, default=0.05)
parser.add_argument('--grasp-height', type=float, default=0.025, help='Minimum box bottom height over table in metres')
parser.add_argument('--grasp-hold', type=float, default=0.2, help='Required continuous grasp duration in seconds')
parser.add_argument('--grasp-force', type=float, default=0.01, help='Minimum box contact force per finger in newtons')
# Perception is isolated from ACT. It only adds four inputs to a NEW residual actor.
parser.add_argument('--vision', choices=('none', 'color-plane'), default='none',
                    help='Residual box estimator; color-plane requires a new residual checkpoint')
parser.add_argument('--vision-config', type=Path,
                    default=Path(__file__).resolve().parents[1] / 'config/vision_color_plane.json',
                    help='HSV/workspace/feature-scale settings for the separate colour estimator')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.vision != 'none':
    if args.mode not in ('act', 'policy', 'residual-ppo'):
        parser.error('--vision requires a policy mode; use vision_cli.py for standalone images')
    try:
        from koch_isaac.vision.color_plane import load_config
        load_config(args.vision_config)
    except (ImportError, OSError, ValueError, KeyError, TypeError) as error:
        parser.error('Vision configuration: ' + str(error))
if args.rerun:
    if args.mode not in ("act", "policy", "residual-ppo"):
        parser.error("--rerun requires --mode policy, act, or residual-ppo")
    if not 0 <= args.rerun_env < args.num_envs:
        parser.error("--rerun-env must be between 0 and num_envs - 1")
    # Fail before launching Isaac if the optional SDK is missing from its Python env.
    import importlib.util
    if importlib.util.find_spec("rerun") is None:
        parser.error("--rerun requires rerun-sdk in the Isaac Lab Python environment")
if args.num_envs > 1 or not args.save_output:
    args.save_output = False
    args.video = False
    args.snapshot = None
    print('Rollout files disabled. Training still saves the residual checkpoint.', flush=True)
if args.train and args.mode != 'residual-ppo': parser.error('--train requires --mode residual-ppo')
if args.checkpoint and args.mode != 'residual-ppo': parser.error('--checkpoint requires --mode residual-ppo')
if args.mode == 'residual-ppo' and not args.train and not args.checkpoint:
    parser.error('Residual evaluation requires --checkpoint')
if args.train and args.steps: parser.error('Use --total-timesteps for training, not --steps')
if args.train_log_every < 0: parser.error('--train-log-every must be nonnegative')
if not 1 <= args.policy_batch_size <= 8: parser.error('--policy-batch-size must be 1..8')
for name in ('residual_limit','grasp_height','grasp_hold','grasp_force','learning_rate'):
    if not 0 < getattr(args,name) < float('inf'): parser.error(name+' must be finite and positive')
if not 0 <= args.residual_penalty < float('inf'): parser.error('residual_penalty must be finite and nonnegative')
if args.train:
    if min(args.total_timesteps,args.ppo_steps,args.ppo_epochs) < 1 or args.ppo_batch_size < 2:
        parser.error('Invalid PPO training size')
    if (args.num_envs*args.ppo_steps) % args.ppo_batch_size:
        parser.error('--ppo-batch-size must divide num_envs * ppo-steps')
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
if args.mode in ("act", "policy", "residual-ppo"):
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
    # Compose scene/actions/observations/rewards/reset/termination settings, then
    # apply CLI overrides BEFORE construction (Isaac creates assets and managers then).
    cfg = KochPickPlaceEnvCfg()
    cfg.scene.num_envs = args.num_envs
    if args.mode in ("act", "policy", "residual-ppo"):
        cfg.num_rerenders_on_reset = 2
        if args.episode_seconds is not None: cfg.episode_length_s = args.episode_seconds
    cfg.sim.use_fabric = args.fabric != "off"
    print(f"Scene synchronization: {'Fabric' if cfg.sim.use_fabric else 'USD'}", flush=True)
    if args.device: cfg.sim.device = args.device
    if args.camera: cfg.enable_front_camera()
    import gymnasium as gym
    from PIL import Image
    if args.mode in ('act','policy','residual-ppo'):
        from koch_isaac.evaluation_env import EvaluatedKochEnv
        # Enables box-filtered normal-force sensors on both jaws. EvaluatedKochEnv
        # captures episode diagnostics before Isaac automatically resets done envs.
        cfg.enable_grasp_evaluation()
        env = EvaluatedKochEnv(cfg, grasp_height=args.grasp_height,
                               grasp_hold=args.grasp_hold, grasp_force=args.grasp_force)
    else:
        env = gym.make("Koch-PinkBox-Place-v0", cfg=cfg).unwrapped
    live_logger = None
    try:
        if args.rerun:
            from koch_isaac.rerun_logger import RerunLogger
            live_logger = RerunLogger(env, args.rerun_env)
            env.rerun_logger = live_logger
            print(f"Rerun live diagnostics: env {args.rerun_env}; select sim_time in the viewer.", flush=True)
        if args.mode in ("act", "policy", "residual-ppo"):
            # Evaluation prints one report per completed episode (up to 600 steps
            # at the default horizon); training also has a separate PPO rollout table.
            if args.train:
                from residual_ppo import train
                train(env,args,calibration)
            else:
                from act_rollout import rollout
                rollout(env,args,calibration)
        else:
            env.reset(seed=args.seed)
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
        try:
            if live_logger is not None:
                live_logger.close()
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
