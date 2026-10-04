"""Scene geometry, contact properties and optional front RGB camera."""
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.utils.configclass import configclass
from .robot_cfg import KOCH_CFG
from . import settings as s


def static_box(name, size, position, color):
    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/" + name,
        init_state=AssetBaseCfg.InitialStateCfg(pos=position),
        spawn=sim_utils.CuboidCfg(
            size=size, collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.6),
        ),
    )


@configclass
class KochSceneCfg(InteractiveSceneCfg):
    robot = KOCH_CFG
    table = static_box("Table", s.TABLE_SIZE, s.TABLE_POS, (0.32, 0.26, 0.20))
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2500.0))
    box = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/PinkBox",
        init_state=RigidObjectCfg.InitialStateCfg(pos=s.BOX_POS),
        spawn=sim_utils.CuboidCfg(
            size=s.BOX_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16, solver_velocity_iteration_count=4,
                max_depenetration_velocity=1.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=s.BOX_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8, dynamic_friction=0.6, restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.16, 0.48)),
        ),
    )
    # Five separate STATIC collision boxes preserve the container's open interior.
    bin_floor = static_box("BinFloor", (s.BIN_INNER[0]+2*s.BIN_WALL, s.BIN_INNER[1]+2*s.BIN_WALL, s.BIN_FLOOR),
                           (s.BIN_POS[0], s.BIN_POS[1], s.BIN_FLOOR/2), (0.95, 0.75, 0.04))
    bin_x1 = static_box("BinX1", (s.BIN_WALL, s.BIN_INNER[1]+2*s.BIN_WALL, s.BIN_INNER[2]),
                        (s.BIN_POS[0]+(s.BIN_INNER[0]+s.BIN_WALL)/2, s.BIN_POS[1], s.BIN_FLOOR+s.BIN_INNER[2]/2), (0.95, 0.75, 0.04))
    bin_x2 = static_box("BinX2", (s.BIN_WALL, s.BIN_INNER[1]+2*s.BIN_WALL, s.BIN_INNER[2]),
                        (s.BIN_POS[0]-(s.BIN_INNER[0]+s.BIN_WALL)/2, s.BIN_POS[1], s.BIN_FLOOR+s.BIN_INNER[2]/2), (0.95, 0.75, 0.04))
    bin_y1 = static_box("BinY1", (s.BIN_INNER[0], s.BIN_WALL, s.BIN_INNER[2]),
                        (s.BIN_POS[0], s.BIN_POS[1]+(s.BIN_INNER[1]+s.BIN_WALL)/2, s.BIN_FLOOR+s.BIN_INNER[2]/2), (0.95, 0.75, 0.04))
    bin_y2 = static_box("BinY2", (s.BIN_INNER[0], s.BIN_WALL, s.BIN_INNER[2]),
                        (s.BIN_POS[0], s.BIN_POS[1]-(s.BIN_INNER[1]+s.BIN_WALL)/2, s.BIN_FLOOR+s.BIN_INNER[2]/2), (0.95, 0.75, 0.04))
    front_camera: CameraCfg | None = None
    static_finger_contact: ContactSensorCfg | None = None
    moving_finger_contact: ContactSensorCfg | None = None


def front_camera_cfg():
    import torch
    from isaaclab.utils.math import quat_from_matrix
    forward = torch.tensor(s.CAMERA_TARGET)-torch.tensor(s.CAMERA_EYE)
    forward = forward / torch.linalg.vector_norm(forward)
    left = torch.linalg.cross(torch.tensor((0.0, 0.0, 1.0)), forward)
    left = left / torch.linalg.vector_norm(left)
    up = torch.linalg.cross(forward, left)
    rotation = tuple(quat_from_matrix(torch.stack((forward, left, up), dim=1)).tolist())
    return CameraCfg(
        offset=CameraCfg.OffsetCfg(pos=s.CAMERA_EYE, rot=rotation, convention="world"),
        prim_path="{ENV_REGEX_NS}/FrontCamera", update_period=1/30,
        height=480, width=640, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, horizontal_aperture=24.0,
                                        clipping_range=(0.01, 1.2)),
    )
