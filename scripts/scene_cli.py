"""Grouped CLI options and pre-launch preparation; safe to import without Isaac.

The public flags/defaults match the original run_scene.py. Keep simulator imports
out of this module so help, validation, and tests do not launch a Kit application.
"""
import argparse
import importlib.util
import json
from pathlib import Path

from utils import DEFAULT_CALIBRATION, encoder_to_sim, lerobot_to_sim, load_calibration

ROOT = Path(__file__).resolve().parents[1]
POLICY_MODES = ("act", "policy", "residual-ppo")


def build_parser():
    """Create project options; the entry point appends Isaac's launcher options."""
    parser = argparse.ArgumentParser(
        description="Preview Koch, evaluate a frozen policy, or train/evaluate residual PPO."
    )
    _add_scene_options(parser)
    _add_joints_options(parser)
    _add_policy_options(parser)
    _add_output_options(parser)
    _add_ppo_options(parser)
    _add_grasp_options(parser)
    _add_vision_options(parser)
    return parser


def _add_scene_options(parser):
    """Choose what to run and how long; Isaac launcher options are added by main()."""
    group = parser.add_argument_group("Scene and execution mode")
    group.add_argument("--num_envs", type=int, default=1)
    group.add_argument("--steps", type=int, default=0, help="0: run until window closes; headless default 300")
    group.add_argument("--mode", choices=("hold", "joints", "act", "policy", "residual-ppo"), default="hold")
    group.add_argument("--camera", action="store_true")
    group.add_argument("--fabric", choices=("auto", "on", "off"), default="auto",
                        help="auto: Fabric enabled for current rendered poses; off is diagnostic only")
    group.add_argument("--check", action="store_true", help="Check state, contacts, reset isolation and success metric")
    group.add_argument("--seed", type=int, default=42)


def _add_joints_options(parser):
    """Only hold mode accepts explicit targets; conversion happens before Kit starts."""
    group = parser.add_argument_group("Joint targets and calibration")
    targets = group.add_mutually_exclusive_group()
    targets.add_argument("--encoder-target", type=float, nargs=6, metavar="COUNT",
                         help="Hold six reported encoder counts, in shoulder pan/lift, elbow, wrist flex/roll, gripper order")
    targets.add_argument("--lerobot-target", type=float, nargs=6, metavar="VALUE",
                         help="Hold six postprocessed LeRobot motor values in the same order")
    group.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    group.add_argument("--lerobot-use-degrees", action="store_true",
                        help="Use only if the training robot used use_degrees=True; gripper remains [0,100]")
    group.add_argument("--print-joints-every", type=int, default=0,
                        help="Print converted measured joints every N steps; 0: final report only")


def _add_policy_options(parser):
    """The policy worker uses its separate Python environment; PPO shares its device."""
    group = parser.add_argument_group("Frozen ACT / diffusion policy")
    group.add_argument("--policy-path", type=Path, help="Local ACT/diffusion pretrained directory; defaults to cached home_v2")
    group.add_argument("--policy-python", type=Path,
                        default=Path("/home/niel/miniforge3/envs/lerobot061/bin/python"))
    group.add_argument("--policy-device", default="cuda", help="Shared ACT/diffusion and PPO device (default: cuda); simulator uses --device")
    group.add_argument('--policy-batch-size', type=int, default=4, help='Frozen policy inference microbatch (1..8)')


def _add_output_options(parser):
    """Episode counts concern evaluation; log intervals are control steps per environment."""
    group = parser.add_argument_group("Evaluation, logging, and live diagnostics")
    group.add_argument("--episodes", type=int, default=5, help="Completed evaluation episodes per environment")
    group.add_argument("--output-dir", type=Path, help="ACT: results directory; default outputs/act_TIMESTAMP")
    group.add_argument("--video", action="store_true", help="ACT: record front.mp4 at 30 simulated fps")
    group.add_argument("--log-every", type=int, default=30, help="ACT: action log interval in steps")
    group.add_argument("--episode-seconds", type=float, help="ACT: override the default 20-second episode")
    group.add_argument('--save-output', action=argparse.BooleanOptionalAction, default=True,
                        help='Save rollout images/logs/video; always disabled when num_envs > 1')
    group.add_argument("--snapshot", type=Path, help="Save front RGB image; implies --camera")
    group.add_argument("--rerun", action="store_true",
                        help="Open live camera/contact/gripper plots in Rerun (policy modes only; no files saved)")
    group.add_argument("--rerun-env", type=int, default=0,
                        help="Environment index shown by --rerun (default: 0)")


def _add_ppo_options(parser):
    """The residual checkpoint is separate from the frozen policy checkpoint."""
    group = parser.add_argument_group("Residual PPO training")
    group.add_argument('--train', action='store_true', help='Train residual PPO instead of evaluating')
    group.add_argument('--checkpoint', type=Path, help='Residual PPO .zip to evaluate or resume')
    group.add_argument('--checkpoint-out', type=Path, help='New trained residual .zip; saved even with --no-save-output')
    group.add_argument('--total-timesteps', type=int, default=100000, help='Training transitions across all environments')
    group.add_argument('--ppo-steps', type=int, default=64, help='Steps per environment in each PPO rollout')
    group.add_argument('--train-log-every', type=int, default=30,
                        help='Extra TensorBoard/console updates every N control steps per env; 0: PPO rollout updates only')
    group.add_argument('--ppo-batch-size', type=int, default=64)
    group.add_argument('--ppo-epochs', type=int, default=4)
    group.add_argument('--learning-rate', type=float, default=3e-4)
    group.add_argument('--residual-limit', type=float, default=0.25, help='Maximum absolute correction per joint in radians')
    group.add_argument('--residual-penalty', type=float, default=0.05)


def _add_grasp_options(parser):
    """These thresholds define reported success; reward terms live in the env config."""
    group = parser.add_argument_group("Grasp evaluation thresholds")
    group.add_argument('--grasp-height', type=float, default=0.025, help='Minimum box bottom height over table in metres')
    group.add_argument('--grasp-hold', type=float, default=0.2, help='Required continuous grasp duration in seconds')
    group.add_argument('--grasp-force', type=float, default=0.01, help='Minimum box contact force per finger in newtons')


def _add_vision_options(parser):
    """Colour-plane features enter only the residual actor, not the frozen ACT."""
    group = parser.add_argument_group("Optional box perception")
    group.add_argument('--vision', choices=('none', 'color-plane'), default='none',
                        help='Residual box estimator; color-plane requires a new residual checkpoint')
    group.add_argument('--vision-config', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'config/vision_color_plane.json',
                        help='HSV/workspace/feature-scale settings for the separate colour estimator')


def _validate_modes(parser, args):
    """Check incompatible options and numeric bounds without starting the simulator."""
    if args.num_envs < 1:
        parser.error("--num_envs must be positive")
    if args.steps < 0:
        parser.error("--steps must be nonnegative")
    if args.print_joints_every < 0:
        parser.error("--print-joints-every must be nonnegative")
    if args.train_log_every < 0:
        parser.error("--train-log-every must be nonnegative")
    if not 1 <= args.policy_batch_size <= 8:
        parser.error("--policy-batch-size must be 1..8")

    if args.train and args.mode != "residual-ppo":
        parser.error("--train requires --mode residual-ppo")
    if args.checkpoint and args.mode != "residual-ppo":
        parser.error("--checkpoint requires --mode residual-ppo")
    if args.mode == "residual-ppo" and not args.train and not args.checkpoint:
        parser.error("Residual evaluation requires --checkpoint")
    if args.train and args.steps:
        parser.error("Use --total-timesteps for training, not --steps")
    if args.mode != "hold" and (
        args.encoder_target is not None or args.lerobot_target is not None
    ):
        parser.error("Explicit joint targets require --mode hold")

    for name in ("residual_limit", "grasp_height", "grasp_hold", "grasp_force", "learning_rate"):
        if not 0 < getattr(args, name) < float("inf"):
            parser.error(name + " must be finite and positive")
    if not 0 <= args.residual_penalty < float("inf"):
        parser.error("residual_penalty must be finite and nonnegative")
    if args.train:
        if min(args.total_timesteps, args.ppo_steps, args.ppo_epochs) < 1 or args.ppo_batch_size < 2:
            parser.error("Invalid PPO training size")
        if (args.num_envs * args.ppo_steps) % args.ppo_batch_size:
            parser.error("--ppo-batch-size must divide num_envs * ppo-steps")

    if args.mode in POLICY_MODES:
        if args.check:
            parser.error("Use --check separately from policy evaluation")
        if args.fabric == "off":
            parser.error("Policies require Fabric for current camera poses; use --fabric on")
        if args.episodes < 1 or args.log_every < 1:
            parser.error("--episodes and --log-every must be positive")
        if args.episode_seconds is not None and not 0 < args.episode_seconds < float("inf"):
            parser.error("--episode-seconds must be finite and positive")
    elif args.video:
        parser.error("--video requires a policy mode")


def _validate_optional_tools(parser, args):
    """Fail early for an unavailable CV backend or Rerun SDK."""
    if args.vision != "none":
        if args.mode not in POLICY_MODES:
            parser.error("--vision requires a policy mode; use vision_cli.py for standalone images")
        try:
            from koch_isaac.vision.color_plane import load_config

            load_config(args.vision_config)
        except (ImportError, OSError, ValueError, KeyError, TypeError) as error:
            parser.error("Vision configuration: " + str(error))

    if args.rerun:
        if args.mode not in POLICY_MODES:
            parser.error("--rerun requires --mode policy, act, or residual-ppo")
        if not 0 <= args.rerun_env < args.num_envs:
            parser.error("--rerun-env must be between 0 and num_envs - 1")
        if importlib.util.find_spec("rerun") is None:
            parser.error("--rerun requires rerun-sdk in the Isaac Lab Python environment")


def _resolve_policy_files(parser, args):
    """Locate the frozen policy locally; no model download or authentication here."""
    if not args.policy_python.is_file():
        parser.error("LeRobot interpreter not found; set --policy-python")
    if args.policy_path is None:
        baseline = json.loads((ROOT / "config/baseline.json").read_text())
        args.policy_path = (
            Path.home() / ".cache/huggingface/hub"
            / ("models--" + baseline["repo_id"].replace("/", "--"))
            / "snapshots" / baseline["cached_revision"]
        )
    args.policy_path = args.policy_path.expanduser().resolve()
    for filename in (
        "config.json", "model.safetensors",
        "policy_preprocessor.json", "policy_postprocessor.json",
    ):
        if not (args.policy_path / filename).is_file():
            parser.error(f"Missing {filename} in checkpoint {args.policy_path}")


def prepare_run(parser, args):
    """Validate options, infer camera requirements, and prepare joint conversion.

    Called BEFORE AppLauncher: malformed commands should fail quickly instead of
    opening a simulator window. Mutates args only to resolve paths and derived
    flags; returns calibration and an optional six-joint target in URDF radians.
    """
    # Multi-env runs suppress rollout files. This does not suppress trained
    # checkpoints or the separately configured TensorBoard training logs.
    if args.num_envs > 1 or not args.save_output:
        args.save_output = False
        args.video = False
        args.snapshot = None
        print("Rollout files disabled. Training still saves the residual checkpoint.", flush=True)

    _validate_modes(parser, args)
    _validate_optional_tools(parser, args)

    try:
        calibration = load_calibration(args.calibration)
        target_radians = None
        if args.encoder_target is not None:
            target_radians = encoder_to_sim(args.encoder_target)
        elif args.lerobot_target is not None:
            target_radians = lerobot_to_sim(
                args.lerobot_target, calibration, use_degrees=args.lerobot_use_degrees
            )
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))

    if args.mode in POLICY_MODES:
        _resolve_policy_files(parser, args)
        args.camera = True
    if args.snapshot:
        args.camera = True
    if args.camera:
        args.enable_cameras = True
    if not args.headless and not args.visualizer:
        args.visualizer = ["kit"]
    return calibration, target_radians
