"""Frozen vision policy plus bounded residual PPO; no robot hardware access."""
from datetime import datetime
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.logger import Logger, HumanOutputFormat
from stable_baselines3.common.callbacks import BaseCallback
from policy_session import PolicySession, ACTOR_DIM, CRITIC_DIM
from run_output import RunOutput

ROOT = Path(__file__).resolve().parents[1]

class AsymmetricExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.latent_dim_pi = self.latent_dim_vf = 128
        self.actor = nn.Sequential(nn.Linear(ACTOR_DIM,128), nn.Tanh(), nn.Linear(128,128), nn.Tanh())
        self.critic = nn.Sequential(nn.Linear(CRITIC_DIM,128), nn.Tanh(), nn.Linear(128,128), nn.Tanh())
    def forward_actor(self, features): return self.actor(features[..., :ACTOR_DIM])
    def forward_critic(self, features): return self.critic(features[..., ACTOR_DIM:])
    def forward(self, features): return self.forward_actor(features), self.forward_critic(features)

class ResidualPolicy(ActorCriticPolicy):
    def _build_mlp_extractor(self):
        self.mlp_extractor = AsymmetricExtractor().to(self.device)
    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        # Deterministic initial correction is exactly zero. PPO exploration is stochastic.
        nn.init.zeros_(self.action_net.weight)
        nn.init.zeros_(self.action_net.bias)

class ResidualVecEnv(VecEnv):
    def __init__(self, session):
        self.session = session
        self.pending = None
        super().__init__(session.num_envs, spaces.Box(-np.inf,np.inf,(ACTOR_DIM+CRITIC_DIM,),np.float32),
                         spaces.Box(-1.,1.,(6,),np.float32))
    def reset(self):
        if self._seeds[0] is not None: self.session.args.seed = self._seeds[0]
        result = self.session.reset()
        self._reset_seeds(); self._reset_options()
        return result
    def step_async(self, actions): self.pending = actions
    def step_wait(self):
        if self.pending is None: raise RuntimeError('step_async required')
        actions, self.pending = self.pending, None
        return self.session.step(actions)
    def close(self): self.session.close()
    def get_attr(self, attr_name, indices=None):
        if attr_name == 'render_mode': return [None for _ in self._get_indices(indices)]
        raise AttributeError(attr_name)
    def set_attr(self, attr_name, value, indices=None):
        raise NotImplementedError('Configure the shared Isaac scene before construction')
    def env_method(self, method_name, *args, indices=None, **kwargs):
        raise NotImplementedError(method_name)
    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._get_indices(indices)]


def validate_contract(model, contract):
    saved = getattr(model, 'residual_contract', None)
    if saved != contract:
        differences = [k for k in contract if saved is None or saved.get(k) != contract[k]]
        raise ValueError('Residual checkpoint does not match this base policy/calibration: '+', '.join(differences))


def load_residual(path, contract, device='cpu', env=None):
    model = PPO.load(str(path), device=device, env=env)
    validate_contract(model, contract)
    return model

class WindowCallback(BaseCallback):
    def __init__(self, env):
        super().__init__()
        self.sim_env = env
    def _on_step(self):
        viewers = self.sim_env.sim.visualizers
        return not viewers or any(v.is_running() and not v.is_closed for v in viewers)


def train(env, args, calibration):
    session = None
    model = None
    status = 'error'
    checkpoint = args.checkpoint_out or ROOT/'checkpoints'/('residual_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.zip')
    checkpoint = checkpoint.expanduser().resolve()
    if checkpoint.suffix != '.zip': checkpoint = checkpoint.with_suffix('.zip')
    # A typo must not overwrite a completed experiment.
    if checkpoint.exists(): raise FileExistsError(f'Checkpoint already exists: {checkpoint}')
    output = RunOutput(args, ROOT, 'residual_train')
    try:
        session = PolicySession(env,args,calibration,output)
        vector = ResidualVecEnv(session)
        if args.checkpoint:
            model = load_residual(args.checkpoint,session.contract,args.policy_device,vector)
        else:
            model = PPO(ResidualPolicy,vector,n_steps=args.ppo_steps,batch_size=args.ppo_batch_size,
                        n_epochs=args.ppo_epochs,learning_rate=args.learning_rate,
                        gamma=0.99,gae_lambda=0.95,clip_range=0.2,target_kl=0.03,
                        policy_kwargs={'log_std_init':-2.0},device=args.policy_device,seed=args.seed,verbose=1)
            model.residual_contract = session.contract
        # No TensorBoard, CSV, monitor file or SB3 default temporary log directory.
        model.set_logger(Logger(folder=None,output_formats=[HumanOutputFormat(sys.stdout)]))
        before = model.num_timesteps
        model.learn(total_timesteps=args.total_timesteps,reset_num_timesteps=not bool(args.checkpoint),
                    callback=WindowCallback(env),progress_bar=True)
        status = 'completed' if model.num_timesteps-before >= args.total_timesteps else 'stopped_early'
    except KeyboardInterrupt:
        status = 'interrupted'
    finally:
        if model is not None and status != 'error':
            checkpoint.parent.mkdir(parents=True,exist_ok=True)
            model.save(str(checkpoint))
            print(f'Residual checkpoint: {checkpoint}',flush=True)
        summary = session.summary(status) if session else {'status':status}
        summary['checkpoint'] = str(checkpoint) if model is not None and status != 'error' else None
        summary['training_only'] = True
        if session: session.close()
        output.finish(summary,session.last_rgb[0] if session and session.last_rgb is not None else None)
