"""Single-environment frozen ACT evaluation with measured state and rendered RGB."""
from datetime import datetime
import json
from pathlib import Path
import re
import time

import numpy as np
import torch
from PIL import Image

from act_bridge import ACTWorker
from utils import SIM_JOINT_NAMES, joint_report, lerobot_to_sim, sim_to_lerobot

ROOT = Path(__file__).resolve().parents[1]


def default_policy_path():
    baseline = json.loads((ROOT / "config/baseline.json").read_text())
    return (Path.home() / ".cache/huggingface/hub" /
            ("models--" + baseline["repo_id"].replace("/", "--")) /
            "snapshots" / baseline["cached_revision"])


def rollout(env, args, calibration):
    if env.num_envs != 1:
        raise ValueError("ACT rollout currently supports one environment")
    if tuple(env.cfg.actions.joints.joint_names) != SIM_JOINT_NAMES:
        raise ValueError("Simulation action order does not match ACT joint order")
    if not np.isclose(env.step_dt, 1/30):
        raise ValueError("This ACT baseline expects 30 Hz simulated control")
    robot = env.scene["robot"]
    joint_ids = [robot.joint_names.index(name) for name in SIM_JOINT_NAMES]
    policy_path = (args.policy_path or default_policy_path()).expanduser().resolve()
    output = args.output_dir or ROOT / "outputs" / ("act_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    log = (output / "rollout.jsonl").open("w")
    video = None
    episodes = []
    total_steps = episode_steps = motor_clipped = sim_clipped = state_clipped = 0
    episode_return = 0.0
    status = "error"
    started = time.monotonic()
    lower, upper = [], []
    for name in SIM_JOINT_NAMES:
        matches = [limits for pattern, limits in env.cfg.actions.joints.clip.items() if re.fullmatch(pattern, name)]
        if len(matches) != 1:
            raise ValueError(f"Expected one configured action limit for {name}")
        lower.append(matches[0][0]); upper.append(matches[0][1])
    motor_low = np.array([-np.inf]*5+[0.] if args.lerobot_use_degrees else [-100.]*5+[0.])
    motor_high = np.array([np.inf]*5+[100.] if args.lerobot_use_degrees else [100.]*6)

    def record(event, **values):
        log.write(json.dumps({"event": event, **values}) + "\n")
        log.flush()

    def read_observation():
        q = robot.data.joint_pos.torch[0, joint_ids].detach().cpu().numpy()
        rgb = env.scene["front_camera"].data.output["rgb"][0, ..., :3].detach().cpu().numpy()
        return q, np.ascontiguousarray(rgb, dtype=np.uint8)

    try:
        if args.video:
            import imageio.v2 as imageio
            video = imageio.get_writer(str(output / "front.mp4"), fps=30, codec="libx264")
        for _ in range(5): env.sim.render()
        _, rgb = read_observation()
        Image.fromarray(rgb).save(output / "first_frame.png")
        print(f"ACT rollout output: {output}", flush=True)
        print("Loading frozen ACT in the LeRobot worker...", flush=True)
        with ACTWorker(args.policy_python, policy_path, args.policy_device, seed=args.seed) as worker:
            record("start", policy_path=str(policy_path), seed=args.seed, control_hz=30,
                   units="degrees" if args.lerobot_use_degrees else "range_m100_100",
                   policy_info={k:v.tolist() for k,v in worker.info.items()},
                   calibration=calibration, initial_joints=joint_report(
                       read_observation()[0], calibration, use_degrees=args.lerobot_use_degrees))
            worker.reset()
            print("ACT ready. Using saved processors and the checkpoint's action chunk queue.", flush=True)
            while len(episodes) < args.episodes and (not args.steps or total_steps < args.steps):
                if env.sim.visualizers and not any(v.is_running() and not v.is_closed for v in env.sim.visualizers):
                    break
                q, rgb = read_observation()
                state = sim_to_lerobot(q, calibration, use_degrees=args.lerobot_use_degrees)
                state_clipped += int(any(joint_report(q, calibration)["outside_calibration_range"]))
                if video is not None: video.append_data(rgb)
                inference_start = time.monotonic()
                raw_action = worker.action(state, rgb)
                inference_seconds = time.monotonic() - inference_start
                motor_clipped += int(np.any((raw_action < motor_low) | (raw_action > motor_high)))
                requested = lerobot_to_sim(raw_action, calibration, use_degrees=args.lerobot_use_degrees)
                applied = np.clip(requested, lower, upper)
                sim_clipped += int(np.any(requested != applied))
                action = torch.as_tensor(applied, dtype=torch.float32, device=env.device).unsqueeze(0)
                with torch.inference_mode():
                    obs, reward, terminated, truncated, extras = env.step(action)
                if not torch.isfinite(reward).all():
                    raise ValueError("Nonfinite simulation reward")
                total_steps += 1
                episode_steps += 1
                episode_return += float(reward[0])
                if total_steps == 1 or total_steps % args.log_every == 0:
                    record("step", step=total_steps, episode=len(episodes)+1,
                           state=state.tolist(), action=raw_action.tolist(),
                           sim_target=applied.tolist(), inference_seconds=inference_seconds)
                if args.print_joints_every and total_steps % args.print_joints_every == 0:
                    print("Joint state: " + json.dumps(joint_report(read_observation()[0], calibration,
                          use_degrees=args.lerobot_use_degrees)), flush=True)
                if bool(terminated[0]) or bool(truncated[0]):
                    outcome = {
                        "episode": len(episodes)+1, "steps": episode_steps,
                        "return": episode_return,
                        "success": bool(env.termination_manager.get_term("success")[0]),
                        "box_lost": bool(env.termination_manager.get_term("box_lost")[0]),
                        "timeout": bool(env.termination_manager.get_term("time_out")[0]),
                    }
                    episodes.append(outcome)
                    record("episode", **outcome)
                    print("ACT episode: " + json.dumps(outcome), flush=True)
                    # env.step has already reset the scene. Never carry an old ACT chunk across it.
                    worker.reset()
                    episode_steps = 0
                    episode_return = 0.0
            status = "completed" if len(episodes) >= args.episodes else "stopped_early"
        _, rgb = read_observation()
        Image.fromarray(rgb).save(output / "last_frame.png")
        if args.snapshot:
            args.snapshot.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgb).save(args.snapshot)
    except KeyboardInterrupt:
        status = "interrupted"
    finally:
        if video is not None: video.close()
        summary = {
            "status": status, "policy_path": str(policy_path),
            "completed_episodes": len(episodes), "requested_episodes": args.episodes,
            "successes": sum(ep["success"] for ep in episodes),
            "success_rate": sum(ep["success"] for ep in episodes)/len(episodes) if episodes else None,
            "steps": total_steps, "partial_episode_steps": episode_steps,
            "motor_clipped_steps": motor_clipped, "sim_clipped_steps": sim_clipped,
            "state_outside_calibration_steps": state_clipped,
            "elapsed_seconds": time.monotonic()-started, "episodes": episodes,
            "note": "Provisional scene/calibration; simulation outcome is not real-robot performance",
        }
        record("summary", **summary)
        log.close()
        (output / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
        print("ACT summary: " + json.dumps(summary), flush=True)
