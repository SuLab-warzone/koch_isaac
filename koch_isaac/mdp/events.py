"""Add EventTermCfg entries here for mass, friction, actuator/camera changes."""
from isaaclab.envs import mdp
from isaaclab.managers import EventTermCfg as EventTerm, SceneEntityCfg
from isaaclab.utils.configclass import configclass


@configclass
class EventsCfg:
    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    # Relative to BOX_POS: front-distance variation, with small lateral variation.
    reset_box = EventTerm(
        func=mdp.reset_root_state_uniform, mode="reset",
        params={"asset_cfg": SceneEntityCfg("box"),
                "pose_range": {"x": (-0.025, 0.025), "y": (-0.015, 0.015), "yaw": (-0.2, 0.2)},
                "velocity_range": {}},
    )
