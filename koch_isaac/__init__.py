"""Import after selecting the Isaac runtime to register the task."""
import gymnasium as gym

gym.register(
    id="Koch-PinkBox-Place-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": "koch_isaac.env_cfg:KochPickPlaceEnvCfg"},
)
