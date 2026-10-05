"""Opt-in Isaac integration check: terminal metrics, timeout state and reset isolation."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--rerun-check', action='store_true',
                    help='Also validate live Rerun data with cameras and an in-memory sink (no viewer)')
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
if args.rerun_check: args.enable_cameras=True
app=AppLauncher(args).app
exit_code=0
try:
    import torch
    from koch_isaac.env_cfg import KochPickPlaceEnvCfg
    from koch_isaac.evaluation_env import EvaluatedKochEnv
    from koch_isaac import settings as s
    from koch_isaac.mdp.observations import box_position
    # Verify reward semantics with deliberately different link origins and COMs.
    # A centered real cuboid alone cannot catch accidentally using root_pos_w.
    from types import SimpleNamespace
    from unittest.mock import patch
    from koch_isaac.mdp import rewards
    from koch_isaac.mdp.observations import box_com_position
    origins=torch.tensor([[0.,0.,0.],[3.,2.,1.]])
    local_com=torch.tensor([[0.1,-0.02,0.004],[0.1,-0.02,0.004]])
    world_com=origins+local_com
    mock_data=SimpleNamespace(root_com_pos_w=SimpleNamespace(torch=world_com),
                              root_pos_w=SimpleNamespace(torch=world_com+torch.tensor([0.,0.,0.04])))
    class MockScene(dict):
        env_origins=origins
    mock_env=SimpleNamespace(scene=MockScene(box=SimpleNamespace(data=mock_data)))
    torch.testing.assert_close(box_com_position(mock_env),local_com)
    with patch.object(rewards,'tcp_position',return_value=local_com):
        torch.testing.assert_close(rewards.reach_box(mock_env),torch.ones(2))
    with patch.object(rewards,'tcp_position',return_value=local_com+torch.tensor([0.,0.,0.04])):
        assert (rewards.reach_box(mock_env)<0.5).all()
    print('PASS: reach reward targets COM rather than link origin and handles env offsets',flush=True)
    cfg=KochPickPlaceEnvCfg(); cfg.scene.num_envs=2
    cfg.enable_grasp_evaluation(); cfg.sim.use_fabric=True
    if args.device: cfg.sim.device=args.device
    if args.rerun_check: cfg.enable_front_camera()
    env=EvaluatedKochEnv(cfg)
    live_logger=None
    try:
        if args.rerun_check:
            import rerun as rr
            from koch_isaac.rerun_logger import RerunLogger
            recording=rr.RecordingStream('koch_rerun_check')
            memory=recording.memory_recording()
            logged_samples=[]
            class CheckedLogger(RerunLogger):
                def log(self, scene, lift):
                    # A zero sample count here would mean we logged AFTER reset.
                    logged_samples.append((scene.common_step_counter,
                                           int(scene.contact_metrics.samples[self.env_id])))
                    super().log(scene, lift)
            live_logger=CheckedLogger(env, env_id=0, recording=recording)
            env.rerun_logger=live_logger
        env.reset(seed=42)
        difference=env.scene['box'].data.root_com_pos_w.torch-env.scene['box'].data.root_pos_w.torch
        print('Box COM minus root [m]: '+str(difference.tolist()),flush=True)
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
            # Forces describe the completed episode, before automatic reset.
            # Env 1 retains its running history while only env 0 is cleared.
            contact = env.terminal_records[0]['contact_forces']
            assert contact['sample_count'] == 31, contact
            assert env.contact_metrics.outcome(0)['sample_count'] == 0
            assert env.contact_metrics.outcome(1)['sample_count'] == 31
            weight = s.BOX_MASS * sum(g*g for g in cfg.sim.gravity) ** 0.5
            assert abs(contact['lift_reference']['minimum_upward_support_N'] - weight) < 1e-8
            for name in ('static_finger_contact', 'moving_finger_contact'):
                assert len(contact[name]['final_normal_force_w_N']) == 3
                assert contact[name]['peak_normal_force_N'] >= contact[name]['final_normal_force_N']
            print('PASS: terminal force report and configured weight; histories reset independently',flush=True)
            assert int(env.episode_length_buf[1])==previous+1
            if args.rerun_check:
                assert len(logged_samples)==31, logged_samples
                assert logged_samples[-1]==(31,31), logged_samples[-1]
                recording.flush(timeout_sec=5)
                assert memory.num_msgs()>0
                print('PASS: Rerun camera/plots serialized; terminal sample logged once before reset',flush=True)
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
            if args.rerun_check:
                assert logged_samples[31]==(32,1), logged_samples[31]
                assert all(step==i+1 and samples>0 for i,(step,samples) in enumerate(logged_samples))
                print('PASS: live timeline stays monotonic across reset; new episode history starts at 1',flush=True)
    finally:
        try:
            if live_logger is not None: live_logger.close()
        finally:
            env.close()
except BaseException:
    import traceback
    traceback.print_exc(); exit_code=1
finally:
    app.close(exit_code=exit_code)
