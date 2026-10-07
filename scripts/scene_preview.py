"""Manual scene preview and diagnostics; no ACT worker or PPO optimizer.

hold   -- keep home joints or an explicitly converted encoder/LeRobot target
joints -- apply small, sequential sinusoidal movements around home
grasp-test -- interpolate user waypoints with contact/lift diagnostics
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
    sequence = args.manual_sequence if args.mode == "grasp-test" else None
    initial = robot.data.joint_pos.torch[0, joint_ids].detach().cpu().numpy().copy()
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
    phase = None
    ended_early = False
    termination_record = None
    for step in range(limit):
        viewers = env.sim.visualizers
        if viewers and not any(v.is_running() and not v.is_closed for v in viewers):
            break

        action = hold.clone()
        if sequence is not None:
            # Command the endpoint on the last transition, then keep it through the hold.
            new_phase, target = sequence.sample((step + 1) * env.step_dt, initial)
            action = torch.as_tensor(target, dtype=hold.dtype, device=env.device).unsqueeze(0)
            if new_phase != phase:
                phase = new_phase
                print("Manual grasp phase: " + json.dumps({
                    "phase": phase, "waypoint_radians": sequence.poses[("open", "align", "close", "lift").index(phase)].tolist()
                }), flush=True)
        elif args.mode == "joints":
            # Exercise one joint every 90 control steps, amplitude 0.15 rad.
            # This is a gentle motion check, not a pick-and-place controller.
            joint = (step // 90) % 6
            action[:, joint] += 0.15 * math.sin((step % 90) / 90 * 2 * math.pi)
        with torch.inference_mode():
            observation, reward, terminated, truncated, _ = env.step(action)
        completed_steps += 1

        if sequence is not None and bool((terminated | truncated).any()):
            # EvaluatedKochEnv captured terminal metrics BEFORE its automatic reset.
            termination_record = env.terminal_records[0]
            ended_early = True
            print("Manual grasp stopped: episode ended before sequence completion", flush=True)
            break

        if args.print_joints_every and completed_steps % args.print_joints_every == 0:
            report = _joint_report(env, joint_ids, calibration, args.lerobot_use_degrees)
            print("Joint state: " + json.dumps(report), flush=True)
            if sequence is not None:
                print("Grasp diagnostics: " + json.dumps({
                    "phase": phase,
                    **env.metrics.outcome(0),
                    "contact_forces": env.contact_metrics.outcome(0),
                }), flush=True)
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
    if sequence is not None:
        record = termination_record or {
            **env.metrics.outcome(0), "contact_forces": env.contact_metrics.outcome(0)
        }
        complete = completed_steps >= math.ceil(sequence.duration / env.step_dt) and not ended_early
        # A historical grasp followed by a drop does not pass the final hold test.
        held_at_end = not ended_early and int(env.metrics.grasp_count[0]) >= env.metrics.grasp_frames
        summary.update(
            mode="grasp-test", phase=phase, sequence_complete=complete,
            grasp=record, final_grasp_hold_s=0.0 if ended_early else int(env.metrics.grasp_count[0]) * env.step_dt,
            status="PASS" if complete and held_at_end else "FAIL" if complete or ended_early else "INCOMPLETE",
            state_after_auto_reset=ended_early,
        )
    print(json.dumps(summary, indent=2))
