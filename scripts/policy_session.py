"""Shared vector simulation loop for frozen base-policy evaluation and residual PPO."""
import json
from pathlib import Path
import re
import time
import numpy as np
import torch
from act_bridge import ACTWorker
from utils import SIM_JOINT_NAMES, ZERO_COUNTS, DIRECTIONS, joint_report, lerobot_to_sim, sim_to_lerobot

# Re-export the old names so existing scripts/checkpoints remain import-compatible.
from residual_observations import (
    ACTOR_DIM, CRITIC_DIM, VISION_DIM,
    make_residual_observation, terminal_residual_observation,
)


class PolicySession:
    """One vector control loop shared by frozen-policy and residual-policy runs.

    _observe: image/joints -> ACT target + optional CV features -> PPO observation.
    step: PPO correction + cached ACT target -> physics -> rewards/reset handling.
    ACT inference lives in its own LeRobot process and never receives PPO gradients.
    """
    def __init__(self, env, args, calibration, output):
        self.env, self.args, self.calibration, self.output = env, args, calibration, output
        self.num_envs = env.num_envs
        self.robot = env.scene["robot"]
        self.joint_ids = [self.robot.joint_names.index(n) for n in SIM_JOINT_NAMES]
        if tuple(env.cfg.actions.joints.joint_names) != SIM_JOINT_NAMES or not np.isclose(env.step_dt,1/30):
            raise ValueError("Unexpected joint order or control frequency")
        limits=[]
        for name in SIM_JOINT_NAMES:
            match=[v for k,v in env.cfg.actions.joints.clip.items() if re.fullmatch(k,name)]
            if len(match)!=1: raise ValueError(f"Ambiguous joint limits: {name}")
            limits.append(match[0])
        self.lower,self.upper=np.asarray(limits).T
        self.previous_residual=np.zeros((self.num_envs,6),dtype=np.float32)
        self.lengths=np.zeros(self.num_envs,dtype=np.int64)
        self.returns=np.zeros(self.num_envs)
        self.completed=[]
        self.steps=0
        self.motor_clipped=self.sim_clipped=self.state_clipped=0
        self.started=time.monotonic()
        self.last_rgb=None
        # Vision is opt-in. Keeping the legacy 25-input actor allows old checkpoints
        # to be evaluated unchanged with --vision none.
        self.vision = None
        self.actor_dim = ACTOR_DIM
        if getattr(args, "vision", "none") == "color-plane":
            from koch_isaac.vision.color_plane import load_config
            from koch_isaac.vision.isaac_adapter import IsaacBoxVision
            self.vision_config = load_config(args.vision_config)
            self.vision = IsaacBoxVision(env, self.vision_config)
            self.actor_dim += VISION_DIM
        self.worker=ACTWorker(args.policy_python,args.policy_path,args.policy_device,
                              seed=args.seed,num_envs=self.num_envs,batch_size=args.policy_batch_size)
        self.contract={"version":1,"policy_path":str(args.policy_path),
                       "policy_type":self.worker.info["policy_type"].item(),
                       "calibration":calibration,"zero_counts":ZERO_COUNTS.tolist(),
                       "directions":DIRECTIONS.tolist(),"use_degrees":args.lerobot_use_degrees,
                       "residual_limit":args.residual_limit,"actor_dim":self.actor_dim,"critic_dim":CRITIC_DIM,
                       "control_hz":30}
        if self.vision is not None:
            from koch_isaac import settings as s
            # Version/config checks prevent loading a legacy or differently scaled
            # actor by accident. Real-camera calibration is independent of this record.
            self.contract.update(version=2, vision={
                "backend": "color-plane", "feature_layout": "box_minus_tcp_xyz_valid_v1",
                "config": self.vision_config, "tcp_offset": list(s.TCP_OFFSET),
                "box_size": list(s.BOX_SIZE), "table_z": s.TABLE_POS[2] + s.TABLE_SIZE[2]/2,
            })
        output.record("start",contract=self.contract,num_envs=self.num_envs,seed=args.seed,
                      grasp_height=args.grasp_height,grasp_hold=args.grasp_hold,grasp_force=args.grasp_force)

    def _observe(self):
        q=self.robot.data.joint_pos.torch[:,self.joint_ids].detach().cpu().numpy()
        velocity=self.robot.data.joint_vel.torch[:,self.joint_ids].detach().cpu().numpy()
        rgb=self.env.scene["front_camera"].data.output["rgb"][...,:3].detach().cpu().numpy()
        self.last_rgb=np.ascontiguousarray(rgb,dtype=np.uint8)
        state=sim_to_lerobot(q,self.calibration,use_degrees=self.args.lerobot_use_degrees)
        report=joint_report(q,self.calibration,use_degrees=self.args.lerobot_use_degrees)
        self.state_clipped+=int(np.count_nonzero(np.any(report["outside_calibration_range"],axis=-1)))
        self.base_values=self.worker.action(state,self.last_rgb)
        lo=np.array([-np.inf]*5+[0.] if self.args.lerobot_use_degrees else [-100.]*5+[0.])
        hi=np.array([np.inf]*5+[100.] if self.args.lerobot_use_degrees else [100.]*6)
        self.motor_clipped+=int(np.count_nonzero(np.any((self.base_values<lo)|(self.base_values>hi),axis=-1)))
        self.base_action=lerobot_to_sim(self.base_values,self.calibration,use_degrees=self.args.lerobot_use_degrees)
        privileged=self.env.observation_manager.compute_group("policy").detach().cpu().numpy()
        vision_features = None
        if self.vision is not None:
            vision_features = self.vision.observe(self.last_rgb, q[:, -1])
            if self.steps % self.args.log_every == 0:
                report = self.vision.report()
                print("Vision: " + json.dumps(report), flush=True)
                self.output.record("vision", step=self.steps, **report)
        self.observation=make_residual_observation(q,velocity,self.base_action,self.previous_residual,
                                                 self.worker.phase,privileged,vision_features)
        return self.observation

    def reset(self):
        self.env.reset(seed=self.args.seed)
        self.worker.reset()
        if self.vision is not None:
            self.vision.reset()
        self.previous_residual[:]=0
        self.lengths[:]=0; self.returns[:]=0
        for _ in range(3): self.env.sim.render()
        return self._observe()

    def step(self, normalized_residual):
        residual=np.asarray(normalized_residual,dtype=np.float32)
        if residual.shape!=(self.num_envs,6) or not np.isfinite(residual).all():
            raise ValueError("Expected finite residual actions [num_envs,6]")
        residual=np.clip(residual,-1,1)
        delta=residual*self.args.residual_limit
        requested=self.base_action+delta
        target=np.clip(requested,self.lower,self.upper)
        if self.vision is not None:
            self.vision.note_action(target)
        self.sim_clipped+=int(np.count_nonzero(np.any(target!=requested,axis=-1)))
        self.output.frame(self.last_rgb[0])
        if self.steps%self.args.log_every==0:
            self.output.record("step",step=self.steps,base_action=self.base_values.tolist(),
                               residual_rad=delta.tolist(),sim_target=target.tolist())
        with torch.inference_mode():
            _,reward,terminated,truncated,_=self.env.step(torch.as_tensor(target,dtype=torch.float32,device=self.env.device))
        reward=reward.detach().cpu().numpy().copy()
        reward-=self.args.residual_penalty*np.square(delta).sum(-1)*self.env.step_dt
        if not np.isfinite(reward).all(): raise ValueError("Nonfinite reward")
        done=(terminated|truncated).cpu().numpy().copy()
        self.steps+=1; self.lengths+=1; self.returns+=reward
        self.previous_residual=residual.copy()
        infos=[{} for _ in range(self.num_envs)]
        for i in np.flatnonzero(done):
            row={"env_id":int(i),"steps":int(self.lengths[i]),"return":float(self.returns[i]),
                 **self.env.terminal_records[int(i)]}
            self.completed.append(row)
            self.output.record("episode",**row)
            infos[i]={**row,"episode":{"r":row["return"],"l":row["steps"]},
                      "TimeLimit.truncated":bool(truncated[i] and not terminated[i]),
                      "terminal_observation":terminal_residual_observation(self.env.terminal_policy_obs[int(i)], self.actor_dim)}
            self.lengths[i]=0; self.returns[i]=0; self.previous_residual[i]=0
        if done.any():
            ids = np.flatnonzero(done)
            self.worker.reset(ids)
            if self.vision is not None:
                self.vision.reset(ids)
        observation=self._observe()
        if self.args.print_joints_every and self.steps%self.args.print_joints_every==0:
            q=self.robot.data.joint_pos.torch[:,self.joint_ids].detach().cpu().numpy()
            print("Joint state: "+json.dumps(joint_report(q,self.calibration,use_degrees=self.args.lerobot_use_degrees)),flush=True)
        return observation,reward,done,infos

    def summary(self,status,episodes=None):
        rows=self.completed if episodes is None else episodes
        n=len(rows)
        return {"status":status,"num_envs":self.num_envs,"vector_steps":self.steps,
                "transitions":self.steps*self.num_envs,"completed_episodes":n,
                "grab_successes":sum(r["grab_success"] for r in rows),
                "grab_success_rate":sum(r["grab_success"] for r in rows)/n if n else None,
                "placement_successes":sum(r["placement_success"] for r in rows),
                "placement_success_rate":sum(r["placement_success"] for r in rows)/n if n else None,
                "pick_place_success_rate":sum(r["pick_place_success"] for r in rows)/n if n else None,
                "timeouts":sum(r["timeout"] for r in rows),"box_lost":sum(r["box_lost"] for r in rows),
                "partial_episode_steps":self.lengths.tolist(),"motor_clipped":self.motor_clipped,
                "sim_clipped":self.sim_clipped,"state_outside_calibration":self.state_clipped,
                "elapsed_seconds":time.monotonic()-self.started,"episodes":rows,
                "vision": self.vision.summary() if self.vision is not None else None,
                "metrics":"Grasp: sustained dual-finger contact and lift; placement: settled/released at episode end"}

    def close(self): self.worker.close()
