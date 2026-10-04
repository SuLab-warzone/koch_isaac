"""Physics integration checks run inside the already launched simulator."""
import torch
from koch_isaac import settings as s
from koch_isaac.mdp.observations import box_position
from koch_isaac.mdp.terminations import placement_candidate


def check_scene(env, home):
    box, robot = env.scene['box'], env.scene['robot']
    assert robot.joint_names == s.JOINT_NAMES, robot.joint_names
    # Gravity and table contact should leave the cube at half its height.
    env.reset(seed=19)
    for _ in range(60): env.step(home)
    assert torch.allclose(box_position(env)[:, 2], torch.full((env.num_envs,), s.BOX_SIZE[2]/2, device=env.device), atol=0.003), box_position(env)
    assert not placement_candidate(env).any(), 'Spawn pose must not count as success'
    print('PASS: box settles on table; initial state is not success')
    # A pan command must physically move the joint without destabilizing the arm.
    action = home.clone(); action[:, 0] += 0.2
    before = robot.data.joint_pos.torch[:, 0].clone()
    for _ in range(60): env.step(action)
    assert ((robot.data.joint_pos.torch[:, 0]-before)>0.12).all(), 'Pan joint failed to follow command'
    print('PASS: position actions move the intended joint')
    env.reset(seed=42)
    if env.num_envs > 1:
        old = box.data.root_state_w.torch[1:].clone()
        env._reset_idx(torch.tensor([0], device=env.device))
        assert torch.allclose(box.data.root_state_w.torch[1:], old), 'Reset leaked to other environments'
        assert box.data.root_vel_w.torch[0].abs().max()<1e-6
        print('PASS: indexed reset is isolated and clears object velocity')
    ids = torch.arange(env.num_envs, device=env.device)
    def put_box(local_pos, velocity=(0.0, 0.0, 0.0)):
        state = box.data.default_root_state.torch.clone()
        state[:, :3] = torch.tensor(local_pos, device=env.device)+env.scene.env_origins
        state[:, 3:7] = torch.tensor([0.,0.,0.,1.], device=env.device)
        state[:, 7:] = 0
        state[:, 7:10] = torch.tensor(velocity, device=env.device)
        box.write_root_state_to_sim(state, env_ids=ids)
        env.scene.update(env.step_dt)
    goal = (s.BIN_POS[0], s.BIN_POS[1], s.BIN_FLOOR+s.BOX_SIZE[2]/2)
    put_box((goal[0], goal[1], goal[2]+0.08))
    assert not placement_candidate(env).any(), 'Hovering is not placement'
    put_box((goal[0]+s.BIN_INNER[0]/2, goal[1], goal[2]))
    assert not placement_candidate(env).any(), 'Overlapping rim is not placement'
    put_box(goal, (0.1,0.,0.))
    assert not placement_candidate(env).any(), 'Moving box is not settled'
    put_box(goal)
    assert placement_candidate(env).all(), 'Resting box should be a placement candidate'
    print('PASS: placement rejects hovering, rim overlap and moving boxes')
    env.termination_manager.reset(ids)
    successes = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for _ in range(24):
        _, _, terminated, _, _ = env.step(home)
        successes |= terminated
    assert successes.all(), 'Stable placement did not terminate successfully'
    print('PASS: bin floor supports box and stable placement triggers success/reset')
