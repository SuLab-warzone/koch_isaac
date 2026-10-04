"""Public Koch follower model. Gains, limits and inertias need calibration."""
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from .settings import URDF, ROOT, JOINT_NAMES, HOME

KOCH_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(URDF), usd_dir=str(ROOT / "assets/koch/usd"),
        fix_base=True, merge_fixed_joints=True,
        collision_type="Convex Decomposition", self_collision=False,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=16,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0), rot=(0.0, 0.0, -0.70710678, 0.70710678),
        joint_pos=dict(zip(JOINT_NAMES, HOME)),
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["follower_joint[1-5]"], stiffness=12.0, damping=0.8,
            effort_limit_sim=2.5, velocity_limit_sim=2.0, armature=0.001,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["follower_joint_gripper"], stiffness=3.0, damping=0.15,
            effort_limit_sim=0.6, velocity_limit_sim=2.0, armature=0.0001,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
