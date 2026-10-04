"""Manager-based RL task: 30 Hz joint commands, 240 Hz physics."""
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup, ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RewTerm, TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass
from isaaclab_physx.physics import PhysxCfg
from . import mdp
from .mdp.events import EventsCfg
from .mdp.curriculum import CurriculumCfg
from .scene_cfg import KochSceneCfg, front_camera_cfg
from .settings import JOINT_NAMES


@configclass
class ActionsCfg:
    # Absolute URDF radians, explicit order: shoulder pan/lift, elbow, wrist flex/roll, gripper.
    joints = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=JOINT_NAMES, preserve_order=True,
        scale=1.0, use_default_offset=False,
        clip={"follower_joint[1-5]": (-2.7, 2.7), "follower_joint_gripper": (0.0, 1.5)},
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        box_position = ObsTerm(func=mdp.box_position)
        box_quat = ObsTerm(func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("box")})
        tcp = ObsTerm(func=mdp.tcp_position)
        goal = ObsTerm(func=mdp.goal_position)
        previous_action = ObsTerm(func=mdp.last_action)
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    policy: PolicyCfg = PolicyCfg()


@configclass
class CameraObservationsCfg(ObservationsCfg):
    @configclass
    class ImagesCfg(ObsGroup):
        front = ObsTerm(func=mdp.front_rgb)
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False
    images: ImagesCfg = ImagesCfg()


@configclass
class RewardsCfg:
    reach = RewTerm(func=mdp.reach_box, weight=1.0)
    lift = RewTerm(func=mdp.lift_box, weight=2.0)
    transport = RewTerm(func=mdp.move_to_bin, weight=3.0)
    placed = RewTerm(func=mdp.placed, weight=10.0)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.002)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    box_lost = DoneTerm(func=mdp.box_lost)
    success = DoneTerm(func=mdp.PlacementSuccess, params={"hold_seconds": 0.5})


@configclass
class KochPickPlaceEnvCfg(ManagerBasedRLEnvCfg):
    scene: KochSceneCfg = KochSceneCfg(num_envs=1, env_spacing=1.5)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventsCfg = EventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    commands = None

    def __post_init__(self):
        self.seed = 42
        self.decimation = 8
        self.episode_length_s = 20.0
        self.sim.dt = 1/240
        self.sim.render_interval = self.decimation
        self.sim.physics = PhysxCfg(bounce_threshold_velocity=0.01)
        self.viewer.eye = (0.7, -0.6, 0.55)
        self.viewer.lookat = (0.15, 0.04, 0.08)

    def enable_front_camera(self):
        self.scene.front_camera = front_camera_cfg()
        self.observations = CameraObservationsCfg()

    def enable_grasp_evaluation(self):
        from isaaclab.sensors import ContactSensorCfg
        from .contact_spawn import spawn_contact_koch
        self.scene.robot.spawn.func = spawn_contact_koch
        self.scene.robot.spawn.activate_contact_sensors = True
        prefix = ("{ENV_REGEX_NS}/Robot/Geometry/follower_base_link/follower_link1_1/"
                  "follower_link2_1/follower_link3_1/follower_link4_1/follower_gripper_static_1")
        self.scene.static_finger_contact = ContactSensorCfg(
            prim_path=prefix, filter_prim_paths_expr=["{ENV_REGEX_NS}/PinkBox"],
            update_period=0.0, history_length=0,
        )
        self.scene.moving_finger_contact = ContactSensorCfg(
            prim_path=prefix+"/follower_gripper_moving_1",
            filter_prim_paths_expr=["{ENV_REGEX_NS}/PinkBox"], update_period=0.0, history_length=0,
        )
        if self.scene.num_envs > 1:
            self.scene.env_spacing = 3.0
