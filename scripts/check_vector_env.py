"""Opt-in Isaac integration check: terminal metrics, timeout state and reset isolation."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
app=AppLauncher(args).app
exit_code=0
try:
    import torch
    from koch_isaac.env_cfg import KochPickPlaceEnvCfg
    from koch_isaac.evaluation_env import EvaluatedKochEnv
    from koch_isaac import settings as s
    from koch_isaac.mdp.observations import box_position
    cfg=KochPickPlaceEnvCfg(); cfg.scene.num_envs=2
    cfg.enable_grasp_evaluation(); cfg.sim.use_fabric=True
    if args.device: cfg.sim.device=args.device
    env=EvaluatedKochEnv(cfg)
    try:
        env.reset(seed=42)
        home=torch.tensor(s.HOME,device=env.device).repeat(2,1)
        with torch.inference_mode():
            for _ in range(30): env.step(home)
            for name in ('static_finger_contact','moving_finger_contact'):
                sensor=env.scene[name]
                assert len(sensor.body_names)==1, sensor.body_names
                forces=sensor.data.force_matrix_w.torch
                assert forces.shape[0]==2 and torch.isfinite(forces).all()
            # Only env 0 times out. Env 1 must advance without resetting.
            env.episode_length_buf[0]=env.max_episode_length-1
            previous=int(env.episode_length_buf[1])
            _,_,terminated,truncated,_=env.step(home)
            assert truncated.tolist()==[True,False]
            assert set(env.terminal_records)=={0}
            assert not env.terminal_records[0]['placement_success']
            assert int(env.episode_length_buf[1])==previous+1
            assert env.terminal_policy_obs[0].shape==(31,)
            assert abs(env.terminal_policy_obs[0][-1]-s.HOME[-1])<1e-6
            assert env.observation_manager.compute_group('policy')[0,-1]==0
            print('PASS: terminal critic input captured BEFORE reset; timeout/reset is isolated',flush=True)
            # Put env 0 at the goal. This tests terminal capture, not policy performance.
            box=env.scene['box']
            state=box.data.default_root_state.torch[:1].clone()
            state[0,:3]=torch.tensor([s.BIN_POS[0],s.BIN_POS[1],s.BIN_FLOOR+s.BOX_SIZE[2]/2],device=env.device)+env.scene.env_origins[0]
            state[0,3:7]=torch.tensor([0.,0.,0.,1.],device=env.device); state[0,7:]=0
            box.write_root_state_to_sim(state,env_ids=torch.tensor([0],device=env.device))
            got_success=False
            for _ in range(30):
                _,_,terminated,_,_=env.step(home)
                if bool(terminated[0]):
                    row=env.terminal_records[0]
                    assert row['placement_success'] and row['success']
                    assert not row['grab_success'] and not row['pick_place_success']
                    assert not env.metrics.outcome(0)['placement_success']
                    got_success=True; break
            assert got_success, 'Placed box failed to produce terminal placement success'
            print('PASS: settled box counts as placement, not grasp; metrics clear after reset',flush=True)
    finally: env.close()
except BaseException:
    import traceback
    traceback.print_exc(); exit_code=1
finally:
    app.close(exit_code=exit_code)
