"""Enable finger contact reports on the imported nested Koch rigid bodies."""
import isaaclab.sim as sim_utils
from pxr import Usd, UsdPhysics

@sim_utils.clone
def spawn_contact_koch(prim_path, cfg, translation=None, orientation=None, **kwargs):
    prim = sim_utils.spawn_from_urdf(prim_path,cfg,translation=translation,orientation=orientation,**kwargs)
    # The generic activation helper stops at the first rigid body. This importer
    # nests articulation links with resetXformStack, so visit the fingers explicitly.
    found = set()
    for body in Usd.PrimRange(prim):
        if body.GetName() in ('follower_gripper_static_1','follower_gripper_moving_1') and body.HasAPI(UsdPhysics.RigidBodyAPI):
            sim_utils.activate_contact_sensors(str(body.GetPath()),stage=body.GetStage())
            found.add(body.GetName())
    if len(found) != 2: raise RuntimeError('Expected two Koch finger rigid bodies')
    return prim
