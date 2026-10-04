import sys
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from act_queue import ActionQueues, ObservationHistory
from run_output import RunOutput
from koch_isaac.metrics import EpisodeMetrics
from residual_ppo import ResidualPolicy, validate_contract
from policy_session import ACTOR_DIM, CRITIC_DIM, terminal_residual_observation
from gymnasium import spaces

class QueueTests(unittest.TestCase):
    def test_selective_reset_preserves_other_chunk(self):
        q=ActionQueues(2,3)
        chunks=np.arange(36,dtype=np.float32).reshape(2,3,6)
        q.put([0,1],chunks)
        np.testing.assert_equal(q.pop([0,1])[0],chunks[:,0])
        q.reset([0]); np.testing.assert_equal(q.hungry([0,1]),[0])
        q.put([0],chunks[:1]+100)
        result,phase=q.pop([0,1])
        np.testing.assert_equal(result[0],chunks[0,0]+100)
        np.testing.assert_equal(result[1],chunks[1,1])
        np.testing.assert_allclose(phase,[0,1/3])
    def test_history_advances_and_resets_per_environment(self):
        h=ObservationHistory(2,2)
        h.append([0,1],np.zeros((2,6)),np.zeros((2,2,2,3),dtype=np.uint8))
        h.append([0,1],np.ones((2,6)),np.ones((2,2,2,3),dtype=np.uint8))
        h.reset([0]); h.append([0,1],np.full((2,6),2),np.full((2,2,2,3),2,dtype=np.uint8))
        state,_=h.batch([0,1])
        np.testing.assert_equal(state[0,:,0],[2,2]); np.testing.assert_equal(state[1,:,0],[1,2])

class MetricTests(unittest.TestCase):
    def test_sustained_grasp_final_placement_and_isolated_reset(self):
        m=EpisodeMetrics(2,'cpu',0.1,0.2,0.2)
        yes=torch.tensor([True,True]); no=torch.tensor([False,False]); lift=torch.tensor([0.03,0.04])
        m.update(yes,yes,lift)
        self.assertFalse(m.outcome(0)['grab_success'])
        m.update(yes,yes,lift)
        self.assertTrue(m.outcome(0)['pick_place_success'])
        m.update(no,no,lift)
        self.assertTrue(m.outcome(0)['grab_success'])
        self.assertFalse(m.outcome(0)['placement_success'])
        m.reset(torch.tensor([0]))
        self.assertFalse(m.outcome(0)['grab_success']); self.assertTrue(m.outcome(1)['grab_success'])

class OutputTests(unittest.TestCase):
    def test_disabled_output_creates_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            for requested,n in [(False,1),(True,2)]:
                args=SimpleNamespace(save_output=requested,num_envs=n,output_dir=Path(d)/'missing',snapshot=Path(d)/'shot.png',video=True)
                out=RunOutput(args,Path(d),'test'); out.frame(np.zeros((2,2,3),dtype=np.uint8))
                out.record('test'); out.finish({'status':'test'},np.zeros((2,2,3),dtype=np.uint8))
                self.assertEqual(list(Path(d).iterdir()),[])

class PolicyTests(unittest.TestCase):
    def test_actor_cannot_read_privileged_state_and_critic_ignores_actor(self):
        policy=ResidualPolicy(spaces.Box(-np.inf,np.inf,(ACTOR_DIM+CRITIC_DIM,),np.float32),
                              spaces.Box(-1.,1.,(6,),np.float32),lambda _:3e-4)
        x=torch.randn(2,ACTOR_DIM+CRITIC_DIM)
        y=x.clone(); y[:,ACTOR_DIM:]+=10
        z=x.clone(); z[:,:ACTOR_DIM]+=10
        with torch.no_grad():
            torch.testing.assert_close(policy.mlp_extractor.forward_actor(x),policy.mlp_extractor.forward_actor(y))
            torch.testing.assert_close(policy.predict_values(x),policy.predict_values(z))
            torch.testing.assert_close(policy.get_distribution(x).distribution.mean,torch.zeros(2,6))
    def test_terminal_value_input_uses_terminal_state(self):
        priv=np.arange(CRITIC_DIM,dtype=np.float32)
        obs=terminal_residual_observation(priv)
        np.testing.assert_equal(obs[ACTOR_DIM:],priv)
    def test_reject_residual_for_different_base(self):
        with self.assertRaises(ValueError):
            validate_contract(SimpleNamespace(residual_contract={'policy_path':'a'}),{'policy_path':'b'})

if __name__=='__main__': unittest.main()
