"""Environment setup and execution routing. Call only after Kit has started.

This layer owns the environment and optional live logger. Policy logic remains
in act_rollout.py/residual_ppo.py; manual joint commands live in scene_preview.py.
"""
from scene_cli import POLICY_MODES


def create_environment(args):
    """Apply configuration overrides BEFORE Isaac creates assets and managers."""
    import gymnasium as gym
    from koch_isaac.env_cfg import KochPickPlaceEnvCfg

    cfg = KochPickPlaceEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.use_fabric = args.fabric != "off"
    if args.device:
        cfg.sim.device = args.device
    if args.camera:
        cfg.enable_front_camera()

    if args.mode in POLICY_MODES + ("grasp-test",):
        # Policy resets need freshly rendered images. Evaluation captures terminal
        # contact/success records before Isaac automatically resets each env.
        cfg.num_rerenders_on_reset = 2
        if args.episode_seconds is not None:
            cfg.episode_length_s = args.episode_seconds
        if args.mode == "grasp-test":
            # A sequence is one uninterrupted attempt, even when longer than 20 s.
            cfg.episode_length_s = args.manual_sequence.duration + 1.0
        cfg.enable_grasp_evaluation()
        from koch_isaac.evaluation_env import EvaluatedKochEnv

        env = EvaluatedKochEnv(
            cfg,
            grasp_height=args.grasp_height,
            grasp_hold=args.grasp_hold,
            grasp_force=args.grasp_force,
        )
    else:
        # Hold/joints mode needs the base task only, without policy processes.
        env = gym.make("Koch-PinkBox-Place-v0", cfg=cfg).unwrapped

    print(f"Scene synchronization: {'Fabric' if cfg.sim.use_fabric else 'USD'}", flush=True)
    return env


def execute_mode(env, args, calibration, target_radians):
    """The three execution paths; each owns its own stepping loop."""
    if args.mode not in POLICY_MODES:
        from scene_preview import run_preview

        run_preview(env, args, calibration, target_radians)
    elif args.train:
        from residual_ppo import train

        # PPO's rollout size controls optimizer updates, not physics frequency.
        train(env, args, calibration)
    else:
        from act_rollout import rollout

        # Plain ACT/diffusion or residual evaluation; reports at episode end.
        rollout(env, args, calibration)


def run_scene(args, calibration, target_radians):
    """Construct, run, and close resources even if a rollout raises an error."""
    env = create_environment(args)
    live_logger = None
    try:
        if args.rerun:
            from koch_isaac.rerun_logger import RerunLogger

            live_logger = RerunLogger(env, args.rerun_env)
            env.rerun_logger = live_logger
            print(
                f"Rerun live diagnostics: env {args.rerun_env}; select sim_time in the viewer.",
                flush=True,
            )
        execute_mode(env, args, calibration, target_radians)
    finally:
        # Closing Rerun must not prevent environment cleanup. run_scene.py closes
        # the outer Kit application after this function returns (or raises).
        try:
            if live_logger is not None:
                live_logger.close()
        finally:
            env.close()
