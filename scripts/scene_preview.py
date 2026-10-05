"""Manual scene preview and diagnostics; no ACT worker or PPO optimizer.

hold   -- keep home joints or an explicitly converted encoder/LeRobot target
joints -- apply small, sequential sinusoidal movements around home
check  -- additionally run finite-state and scene smoke checks
"""
import json
import math

import torch

from utils import SIM_JOINT_NAMES, joint_report


def _joint_report(env, joint_ids, calibration, use_degrees):
    positions = env.scene["robot"].data.joint_pos.torch[0, joint_ids]
    return joint_report(positions.detach().cpu().numpy(), calibration, use_degrees=use_degrees)


def _save_snapshot(env, path):
    from PIL import Image

    # A few explicit renders make a final snapshot current even after stepping stops.
    for _ in range(5):
        env.sim.render()
    rgb = env.scene["front_camera"].data.output["rgb"][0, ..., :3].cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path)
    print(f"Saved camera: {path}", flush=True)


def run_preview(env, args, calibration, target_radians):
    """Step absolute joint-position targets at the environment's control rate."""
    from koch_isaac import settings as settings

    env.reset(seed=args.seed)
    if tuple(settings.JOINT_NAMES) != SIM_JOINT_NAMES:
        raise ValueError("Action joint order differs from the calibration mapping")
    robot = env.scene["robot"]
    joint_ids = [robot.joint_names.index(name) for name in SIM_JOINT_NAMES]
    home = torch.tensor(settings.HOME, device=env.device).repeat(env.num_envs, 1)
    hold = home
    if target_radians is not None:
        hold = torch.tensor(
            target_radians, dtype=home.dtype, device=env.device
        ).repeat(env.num_envs, 1)

    target_report = joint_report(
        hold[0].cpu().numpy(), calibration, use_degrees=args.lerobot_use_degrees
    )
    print("Joint target: " + json.dumps(target_report), flush=True)
    print(
        "Targets pass through the environment's configured joint limits; calibration is provisional.",
        flush=True,
    )

    # GUI mode normally runs until closed; headless preview defaults to 300 steps.
    limit = args.steps or (300 if args.headless else 10**9)
    completed_steps = 0
    observation = None
    for step in range(limit):
        viewers = env.sim.visualizers
        if viewers and not any(v.is_running() and not v.is_closed for v in viewers):
            break

        action = hold.clone()
        if args.mode == "joints":
            # Exercise one joint every 90 control steps, amplitude 0.15 rad.
            # This is a gentle motion check, not a pick-and-place controller.
            joint = (step // 90) % 6
            action[:, joint] += 0.15 * math.sin((step % 90) / 90 * 2 * math.pi)
        with torch.inference_mode():
            observation, reward, _, _, _ = env.step(action)
        completed_steps += 1

        if args.print_joints_every and completed_steps % args.print_joints_every == 0:
            report = _joint_report(env, joint_ids, calibration, args.lerobot_use_degrees)
            print("Joint state: " + json.dumps(report), flush=True)
        if args.check:
            assert torch.isfinite(observation["policy"]).all(), "Non-finite observation"
            assert torch.isfinite(reward).all(), "Non-finite reward"

    if args.snapshot:
        _save_snapshot(env, args.snapshot)
    if args.check:
        from smoke_checks import check_scene

        check_scene(env, home)

    # The viewer can close before the first step: avoid reading an unset observation.
    summary = {
        "steps": completed_steps,
        "num_envs": env.num_envs,
        "joint_names": robot.joint_names,
        "joint_positions": robot.data.joint_pos.torch[0].tolist(),
        "joint_conversion": _joint_report(env, joint_ids, calibration, args.lerobot_use_degrees),
        "box_local": (
            env.scene["box"].data.root_pos_w.torch - env.scene.env_origins
        )[0].tolist(),
        "observation_shape": list(observation["policy"].shape) if observation is not None else None,
        "use_fabric": env.cfg.sim.use_fabric,
        "device": str(env.device),
        "status": "PASS" if completed_steps else "stopped_before_first_step",
    }
    print(json.dumps(summary, indent=2))
