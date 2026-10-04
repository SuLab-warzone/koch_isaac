"""Curriculum hook. Empty by default so the initial scene is easy to debug.

Example for a later stage:
    from isaaclab.managers import CurriculumTermCfg as CurrTerm
    from isaaclab.envs.mdp import modify_reward_weight
    reach = CurrTerm(func=modify_reward_weight,
        params={"term_name": "reach", "weight": 0.1, "num_steps": 100000})

Expand the reset_box pose_range only after checking reachability and collisions.
"""
from isaaclab.utils.configclass import configclass

@configclass
class CurriculumCfg:
    pass
